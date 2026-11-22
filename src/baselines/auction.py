import time
from typing import List, Dict, Any
from src.env.entities import Robot, Task
from src.communication.channel import CommunicationChannel
from config import DEFAULT_CONFIG
from src.baselines.sequencing import sequence_feasible


class AuctionAllocator:
    """
    Sequential Single-Item (SSI) auction with a logical auctioneer.

    For L1-L4, eligible robots transmit one BID message to the auctioneer.
    Only bids actually delivered through CommunicationChannel are considered.
    L0 has no bid channel; eligible robots contend locally and the simulator
    applies the documented deterministic arbitration rule (lowest estimated
    cost, then lowest robot ID), recording losing claims as conflicts.
    """
    def __init__(self, comm_channel: CommunicationChannel, config=DEFAULT_CONFIG, sequence_aware: bool = True):
        self.channel = comm_channel
        self.config = config
        self.sequence_aware = sequence_aware

    def solve(self, robots: List[Robot], tasks: List[Task]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self.channel.reset()

        assignments: Dict[int, List[int]] = {r.id: [] for r in robots}
        remaining_caps = {r.id: r.capacity_limit for r in robots}
        unassigned_task_ids = [t.id for t in tasks]
        task_dict = {t.id: t for t in tasks}

        conflict_count = 0
        total_assignment_cost = 0.0
        current_step = 0
        max_steps = self.config.max_steps
        is_no_comm = self.channel.k == float("inf")

        while unassigned_task_ids and current_step < max_steps:
            target_task_id = unassigned_task_ids[0]
            target_task = task_dict[target_task_id]

            eligible = []
            for robot in robots:
                if remaining_caps[robot.id] <= 0:
                    continue
                if target_task.capability_req not in robot.capabilities:
                    continue
                # Deadline feasibility is judged from the execution clock t = 0 (planning happens
                # before execution), not from the auction-round counter.
                if self.sequence_aware:
                    if not sequence_feasible(robot, task_dict, assignments[robot.id], target_task, self.config):
                        continue
                elif robot.estimated_completion_time(target_task, 0) > target_task.deadline:
                    continue
                bid_cost = robot.estimated_cost_to_task(
                    target_task, self.config.alpha, self.config.beta
                )
                eligible.append((float(bid_cost), robot.id))

            if is_no_comm:
                # No bids can be transmitted. Robots make independent local claims;
                # the simulator resolves the physical contention deterministically.
                received_bids = list(eligible)
                if len(received_bids) > 1:
                    conflict_count += len(received_bids) - 1
            else:
                # Auctions are cleared only at a synchronization point. The
                # coordinator considers only successfully transmitted, delivered bids.
                k = self.channel.k
                if k not in (0, float("inf")) and current_step % k:
                    current_step += k - (current_step % k)
                if current_step >= max_steps:
                    break

                for bid_cost, robot_id in eligible:
                    self.channel.send_to_coordinator(
                        sender_id=robot_id,
                        msg_type="BID",
                        payload={"task_id": target_task_id, "bid": bid_cost},
                        current_step=current_step
                    )

                # Deliver any queued messages at this synchronization point.
                self.channel.flush_buffer_if_period(current_step)
                delivered = self.channel.receive_coordinator_messages()
                received_bids = [
                    (float(msg.payload["bid"]), msg.sender_id)
                    for msg in delivered
                    if msg.msg_type == "BID"
                    and msg.payload.get("task_id") == target_task_id
                ]

            if received_bids:
                received_bids.sort(key=lambda item: (item[0], item[1]))
                winner_cost, winner_id = received_bids[0]
                assignments[winner_id].append(target_task_id)
                remaining_caps[winner_id] -= 1
                total_assignment_cost += winner_cost
            # If no bid is received, this task remains unassigned.
            unassigned_task_ids.pop(0)
            current_step += 1

        all_assigned = {
            task_id for assigned_list in assignments.values() for task_id in assigned_list
        }
        unassigned = [task.id for task in tasks if task.id not in all_assigned]
        unassigned_penalty = len(unassigned) * self.config.penalty_unassigned
        conflict_penalty = conflict_count * self.config.penalty_conflict

        # The static assignment objective used for ILP comparison excludes
        # communication/conflict overhead. Conflict penalty is reported separately.
        assignment_objective_cost = total_assignment_cost + unassigned_penalty
        total_cost = assignment_objective_cost + conflict_penalty
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "total_cost": total_cost,
            "assignment_cost": total_assignment_cost,
            "unassigned_penalty": unassigned_penalty,
            "conflict_penalty": conflict_penalty,
            "assignment_objective_cost": assignment_objective_cost,
            "assignments": assignments,
            "unassigned_tasks": unassigned,
            "conflict_count": conflict_count,
            "messages_sent": self.channel.total_messages,
            "assignment_rate": len(all_assigned) / len(tasks) if tasks else 0.0,
            "completion_rate": len(all_assigned) / len(tasks) if tasks else 0.0,
            "runtime_ms": elapsed_ms
        }
