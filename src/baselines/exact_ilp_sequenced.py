"""
Sequence-aware exact reference (set-partitioning ILP).

Why this file exists
--------------------
`exact_ilp.py` solves the assignment model of the proposal (Section 3): every
x_ij is checked against its deadline *as if robot i started task j at t = 0
from its initial position*. A robot that receives 2-3 tasks must however do
them one after another, so the later tasks finish much later than that model
assumes. The plan of the assignment ILP is therefore optimal only for an
optimistic model; replayed in the warehouse simulator it misses many deadlines.

This module removes that optimism while staying an *exact* integer program.
For N = 3 robots, K = 3 tasks per robot and M = 10 tasks the number of possible
ordered task sequences per robot is tiny (at most 10 + 90 + 720), so we
enumerate them as columns:

    column (i, s) = robot i serves the ordered sequence s = (j1, ..., jL), L <= K_i

A column is kept only if
  * every task in s matches robot i's capabilities,
  * every task in s is delivered no later than its deadline when the tasks are
    executed back-to-back by the simulator's movement rules, and
  * the travelled distance fits in the battery.

Its cost is the travel distance (the physical cost of the simulator). Then

    min   sum_(i,s) travel(i,s) * y_(i,s)  +  P_u * sum_j u_j
    s.t.  sum_(i,s: j in s) y_(i,s) + u_j = 1     for every task j
          sum_s y_(i,s) <= 1                      for every robot i
          y, u binary

Because every selected column is on time by construction, the optimum equals the
best achievable *physical* cost (travel + P_u * unassigned) and replaying the plan
in `WarehouseMRTAEnv.run_physical_execution` reproduces exactly this cost with
zero missed deadlines (checked in tests/test_sequenced_reference.py).
"""
import itertools
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from config import DEFAULT_CONFIG
from src.env.entities import Robot, Task


def _leg_steps(distance: int) -> int:
    """The simulator needs one step even for a zero-length leg (phase change)."""
    return distance if distance > 0 else 1


class SequenceAwareExactSolver:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def _columns(self, robots: List[Robot], tasks: List[Task]) -> List[Tuple[int, Tuple[int, ...], int]]:
        columns = []
        for robot in robots:
            usable = [t for t in tasks if t.capability_req in robot.capabilities]
            for length in range(1, robot.capacity_limit + 1):
                for seq in itertools.permutations(usable, length):
                    position, clock, travel, feasible = robot.position, 0, 0, True
                    for task in seq:
                        d_pick = position.manhattan_distance(task.pickup)
                        d_del = task.travel_distance
                        clock += _leg_steps(d_pick) + _leg_steps(d_del) + task.duration
                        travel += d_pick + d_del
                        if clock > task.deadline:
                            feasible = False
                            break
                        position = task.delivery
                    energy = travel * self.config.battery_cost_per_unit_dist
                    if feasible and energy <= robot.battery + 1e-9:
                        columns.append((robot.id, tuple(t.id for t in seq), travel))
        return columns

    def solve(self, robots: List[Robot], tasks: List[Task]) -> Dict[str, Any]:
        start = time.perf_counter()
        n_robots, n_tasks = len(robots), len(tasks)
        task_index = {t.id: j for j, t in enumerate(tasks)}
        robot_index = {r.id: i for i, r in enumerate(robots)}
        columns = self._columns(robots, tasks)
        n_cols = len(columns)
        n_vars = n_cols + n_tasks            # y columns followed by u_j

        cost = np.array([c[2] for c in columns] + [self.config.penalty_unassigned] * n_tasks, dtype=float)
        a = np.zeros((n_tasks + n_robots, n_vars))
        lo = np.zeros(n_tasks + n_robots)
        hi = np.zeros(n_tasks + n_robots)
        for k, (rid, seq, _) in enumerate(columns):
            for tid in seq:
                a[task_index[tid], k] = 1.0
            a[n_tasks + robot_index[rid], k] = 1.0
        for j in range(n_tasks):
            a[j, n_cols + j] = 1.0
            lo[j] = hi[j] = 1.0
        hi[n_tasks:] = 1.0                    # at most one column per robot

        result = milp(
            c=cost,
            integrality=np.ones(n_vars),
            constraints=LinearConstraint(a, lo, hi),
            bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        )
        if not result.success:
            raise RuntimeError(f"Sequence-aware ILP failed: {result.message}")

        assignments: Dict[int, List[int]] = {r.id: [] for r in robots}
        travel = 0
        for k, (rid, seq, col_travel) in enumerate(columns):
            if result.x[k] > 0.5:
                assignments[rid] = list(seq)
                travel += col_travel
        assigned = {tid for seq in assignments.values() for tid in seq}
        unassigned = [t.id for t in tasks if t.id not in assigned]
        total_cost = travel + len(unassigned) * self.config.penalty_unassigned
        return {
            "success": True,
            "total_cost": float(total_cost),
            "optimal_cost": float(total_cost),
            "travel_distance": float(travel),
            "assignments": assignments,
            "unassigned_tasks": unassigned,
            "assignment_rate": len(assigned) / n_tasks if n_tasks else 0.0,
            "n_columns": n_cols,
            "runtime_ms": (time.perf_counter() - start) * 1000.0,
        }
