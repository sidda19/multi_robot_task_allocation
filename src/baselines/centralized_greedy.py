import time
from typing import List, Dict, Any, Optional
from src.env.entities import Robot, Task
from config import DEFAULT_CONFIG
from src.baselines.sequencing import sequence_feasible

class CentralizedGreedyAllocator:
    """
    Centralized Greedy Baseline (Faithful to Ideation Report Section 4.1):
    Full-information centralized reference.
    
    Mathematical Formulation (Ideation Report Eq. 14-15):
    For each task T_j (in deterministic sequence j = 0 ... M-1):
        F_j = {i : q_j in Q_i, R_i satisfies resource constraints (capacity and deadline)}
        If F_j is non-empty:
            i* = argmin_{i in F_j} c_ij
            Assign task T_j to robot i*
        Else:
            Leave task T_j unassigned.
    """
    def __init__(self, config=DEFAULT_CONFIG, task_order_policy: str = "natural", sequence_aware: bool = True):
        """
        task_order_policy:
            'natural': Tasks are processed in deterministic index order T_0 ... T_{M-1}.
            'deadline': Tasks are processed in ascending order of deadlines.
        """
        self.config = config
        self.sequence_aware = sequence_aware
        self.task_order_policy = task_order_policy

    def solve(self, robots: List[Robot], tasks: List[Task]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        tasks_by_id = {t.id: t for t in tasks}
        remaining_caps = {r.id: r.capacity_limit for r in robots}
        assignments: Dict[int, List[int]] = {r.id: [] for r in robots}
        assigned_tasks = set()
        total_assignment_cost = 0.0

        # Deterministic task ordering policy
        if self.task_order_policy == "deadline":
            ordered_tasks = sorted(tasks, key=lambda t: (t.deadline, t.id))
        else:
            # Default natural task order T_0, T_1, ..., T_{M-1} as formulated
            ordered_tasks = sorted(tasks, key=lambda t: t.id)

        # Task-wise selection: for each task T_j
        for t in ordered_tasks:
            # Construct feasible set F_j: robots satisfying capability, capacity, and deadline
            feasible_candidates = []
            for r in robots:
                # 1. Capability feasibility
                if t.capability_req in r.capabilities:
                    # 2. Capacity constraint
                    if remaining_caps[r.id] > 0:
                        # 3. Deadline feasibility
                        if self.sequence_aware:
                            ok = sequence_feasible(r, tasks_by_id, assignments[r.id], t, self.config)
                        else:
                            ok = r.estimated_completion_time(t, current_step=0) <= t.deadline
                        if ok:
                            cost = r.estimated_cost_to_task(t, self.config.alpha, self.config.beta)
                            feasible_candidates.append((cost, r.id))

            # If F_j is non-empty, select robot with minimum assignment cost:
            # i* = argmin_{i in F_j} c_ij (tie-breaking with lower robot ID)
            if feasible_candidates:
                feasible_candidates.sort(key=lambda x: (x[0], x[1]))
                best_cost, best_robot_id = feasible_candidates[0]
                
                assignments[best_robot_id].append(t.id)
                assigned_tasks.add(t.id)
                remaining_caps[best_robot_id] -= 1
                total_assignment_cost += best_cost

        unassigned = [t.id for t in tasks if t.id not in assigned_tasks]
        unassigned_penalty = len(unassigned) * self.config.penalty_unassigned
        total_cost = total_assignment_cost + unassigned_penalty
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "total_cost": total_cost,
            "assignment_objective_cost": total_cost,
            "assignment_cost": total_assignment_cost,
            "unassigned_penalty": unassigned_penalty,
            "conflict_penalty": 0.0,
            "assignments": assignments,
            "unassigned_tasks": unassigned,
            "conflict_count": 0,
            "messages_sent": 0,
            "assignment_rate": len(assigned_tasks) / len(tasks) if tasks else 0.0,
            "completion_rate": len(assigned_tasks) / len(tasks) if tasks else 0.0,
            "runtime_ms": elapsed_ms
        }
