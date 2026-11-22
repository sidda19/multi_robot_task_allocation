import time
from typing import List, Dict, Set, Any, Tuple, Optional
from src.env.entities import Robot, Task
from src.communication.channel import CommunicationChannel
from config import DEFAULT_CONFIG
from src.baselines.sequencing import sequence_feasible

class DistributedGreedyAllocator:
    """
    Distributed Greedy Allocation:
    Decentralized decision-making where each robot maintains its own local beliefs
    and coordinates solely through messages received via CommunicationChannel (L0-L4).

    Rules:
    - No hidden information updates: robots learn teammate claims ONLY via received messages.
    - When a robot claims a task, it broadcasts a CLAIM message.
    - If multiple robots claim the same task, the contender with the lower estimated assignment
      cost keeps the task; robot ID is used solely as a deterministic tie-breaker for equal costs.
    - A losing robot yields and reconsiders alternative tasks ONLY after receiving the teammate's
      CLAIM message through the channel according to the communication schedule (k, B).
    - Under L0 (no communication), no messages are delivered; conflicts remain unresolved.
    """
    def __init__(self, comm_channel: CommunicationChannel, config=DEFAULT_CONFIG, sequence_aware: bool = True):
        self.channel = comm_channel
        self.config = config
        self.sequence_aware = sequence_aware

    def solve(self, robots: List[Robot], tasks: List[Task]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self.channel.reset()

        task_dict = {t.id: t for t in tasks}
        robot_dict = {r.id: r for r in robots}

        # Strictly local state for each robot:
        # local_known_claimed: tasks that robot knows are claimed by teammates (learned via received messages)
        local_known_claimed: Dict[int, Set[int]] = {r.id: set() for r in robots}
        # local_assignments: tasks currently claimed by this robot
        local_assignments: Dict[int, List[int]] = {r.id: [] for r in robots}
        # task_costs: estimated cost of locally claimed tasks
        task_costs: Dict[int, Dict[int, float]] = {r.id: {} for r in robots}
        # remaining capacity for each robot
        remaining_caps: Dict[int, int] = {r.id: r.capacity_limit for r in robots}

        # A conflict is counted once per losing robot-task pair, even if it is
        # observed both during simultaneous claiming and final reconciliation.
        conflict_pairs: Set[Tuple[int, int]] = set()  # (task_id, losing_robot_id)

        current_step = 0
        max_steps = self.config.max_steps

        while current_step < max_steps:
            # 1. Flush channel buffer if this is a communication synchronization step
            self.channel.flush_buffer_if_period(current_step)

            # 2. Receive and process messages from channel for each robot
            for r in robots:
                inbox = self.channel.receive_messages(r.id)
                for msg in inbox:
                    if msg.msg_type == "CLAIM":
                        t_id = msg.payload.get("task_id")
                        sender_cost = msg.payload.get("cost", float("inf"))
                        sender_id = msg.sender_id

                        if t_id is not None:
                            if t_id in local_assignments[r.id]:
                                # Both robot r.id and sender_id claim task t_id!
                                my_cost = task_costs[r.id][t_id]
                                # Conflict resolution rule: lower cost wins; tie-break on lower robot ID
                                if (sender_cost, sender_id) < (my_cost, r.id):
                                    # Teammate has better claim: I yield
                                    local_assignments[r.id].remove(t_id)
                                    remaining_caps[r.id] += 1
                                    del task_costs[r.id][t_id]
                                    local_known_claimed[r.id].add(t_id)
                                    # Now robot r.id has capacity and knows t_id is taken
                                else:
                                    # My claim is superior or equal with lower ID: I keep it
                                    pass
                            else:
                                # Teammate claimed a task I do not hold; record in local belief
                                local_known_claimed[r.id].add(t_id)

            # 3. Independent local selection for each robot with remaining capacity
            newly_claimed_this_step: Dict[int, List[int]] = {} # t_id -> list of claiming robot_ids

            for r in robots:
                if remaining_caps[r.id] > 0 and len(local_assignments[r.id]) < r.capacity_limit:
                    best_task_id = None
                    best_cost = float("inf")

                    for t_id, t in task_dict.items():
                        # Consider only tasks not in local belief of claimed tasks and not already held
                        if t_id not in local_known_claimed[r.id] and t_id not in local_assignments[r.id]:
                            if t.capability_req in r.capabilities:
                                # Allocation is planned before execution starts, so deadline feasibility is
                                # judged from the execution clock t = 0, not from the message-round counter.
                                if self.sequence_aware:
                                    ok = sequence_feasible(r, task_dict, local_assignments[r.id], t, self.config)
                                else:
                                    ok = r.estimated_completion_time(t, 0) <= t.deadline
                                if ok:
                                    cost = r.estimated_cost_to_task(t, self.config.alpha, self.config.beta)
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_task_id = t_id

                    if best_task_id is not None:
                        # Robot r claims best_task_id locally
                        local_assignments[r.id].append(best_task_id)
                        task_costs[r.id][best_task_id] = best_cost
                        remaining_caps[r.id] -= 1

                        if best_task_id not in newly_claimed_this_step:
                            newly_claimed_this_step[best_task_id] = []
                        newly_claimed_this_step[best_task_id].append(r.id)

                        # Broadcast claim to teammates (subject to channel timing & budget)
                        self.channel.broadcast(
                            sender_id=r.id,
                            msg_type="CLAIM",
                            payload={"task_id": best_task_id, "cost": best_cost},
                            current_step=current_step
                        )

            # 4. Check for simultaneous claims in this step
            for t_id, claimants in newly_claimed_this_step.items():
                if len(claimants) > 1:
                    task = task_dict[t_id]
                    ordered = sorted(
                        claimants,
                        key=lambda rid: (task_costs[rid][t_id], rid)
                    )
                    for loser_id in ordered[1:]:
                        conflict_pairs.add((t_id, loser_id))

            # Check termination: all tasks known claimed or all robots at capacity
            all_known = all(
                len(local_known_claimed[r.id]) + len(local_assignments[r.id]) >= len(tasks)
                or remaining_caps[r.id] == 0
                for r in robots
            )
            no_new_claims = len(newly_claimed_this_step) == 0

            # If no robot can make new claims and no messages are pending in outbox
            if no_new_claims and len(self.channel.outbox_buffer) == 0:
                break

            current_step += 1

        # 5. Final Ground Truth Ownership Reconciliation
        # In the physical warehouse, each task can have at most ONE owner.
        # Contenders are resolved by lower estimated cost, then lower robot ID.
        final_assignments: Dict[int, List[int]] = {r.id: [] for r in robots}
        total_assignment_cost = 0.0

        winners: Dict[int, int] = {}
        for t in tasks:
            # Find all robots that believe they own task t.id
            claimants = [r.id for r in robots if t.id in local_assignments[r.id]]
            if len(claimants) == 1:
                winners[t.id] = claimants[0]
            elif len(claimants) > 1:
                # Unresolved conflict (e.g. under L0 or before sync):
                # winner is lower cost, tie-break lower robot ID.
                sorted_claimants = sorted(claimants, key=lambda r_id: (task_costs[r_id][t.id], r_id))
                winners[t.id] = sorted_claimants[0]
                # Record any conflict not already observed during simultaneous claims.
                for loser_id in sorted_claimants[1:]:
                    conflict_pairs.add((t.id, loser_id))
        # Keep every robot's CLAIM ORDER: the deadline check was made for that route order.
        for r in robots:
            # legacy (sequence_aware=False) behaviour: tasks in task-id order
            ordered_ids = local_assignments[r.id] if self.sequence_aware else sorted(local_assignments[r.id])
            for t_id in ordered_ids:
                if winners.get(t_id) == r.id:
                    final_assignments[r.id].append(t_id)
                    total_assignment_cost += task_costs[r.id][t_id]

        all_assigned = set()
        for assigned_list in final_assignments.values():
            all_assigned.update(assigned_list)

        unassigned = [t.id for t in tasks if t.id not in all_assigned]
        unassigned_penalty = len(unassigned) * self.config.penalty_unassigned
        conflict_count = len(conflict_pairs)
        conflict_penalty = conflict_count * self.config.penalty_conflict
        total_cost = total_assignment_cost + unassigned_penalty + conflict_penalty

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "total_cost": total_cost,
            "assignment_objective_cost": total_assignment_cost + unassigned_penalty,
            "assignment_cost": total_assignment_cost,
            "unassigned_penalty": unassigned_penalty,
            "conflict_penalty": conflict_penalty,
            "assignments": final_assignments,
            "unassigned_tasks": unassigned,
            "conflict_count": conflict_count,
            "messages_sent": self.channel.total_messages,
            "assignment_rate": len(all_assigned) / len(tasks) if tasks else 0.0,
            "completion_rate": len(all_assigned) / len(tasks) if tasks else 0.0, # Kept for backward compatibility
            "runtime_ms": elapsed_ms
        }
