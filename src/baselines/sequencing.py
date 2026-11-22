"""Order-aware deadline/battery check shared by the static planners.

A robot that already holds tasks s_1..s_L must serve them one after another, so a
new task appended behind them finishes much later than "from the start position
at t = 0". `sequence_feasible` checks the new task against the ROBOT'S WHOLE
ROUTE using the simulator's movement rules (a zero-length leg costs one step).
Appending a task never delays the tasks already in the route, so only the new
task needs to be checked.
"""
from typing import Dict, List, Tuple

from src.baselines.exact_ilp_sequenced import _leg_steps


def route_finish_and_travel(robot, tasks_by_id: Dict[int, object], seq_ids: List[int], new_task) -> Tuple[int, int]:
    pos = robot.position
    clock = 0
    travel = 0
    for task in [tasks_by_id[j] for j in seq_ids] + [new_task]:
        d_pick = pos.manhattan_distance(task.pickup)
        d_del = task.travel_distance
        clock += _leg_steps(d_pick) + _leg_steps(d_del) + task.duration
        travel += d_pick + d_del
        pos = task.delivery
    return clock, travel


def sequence_feasible(robot, tasks_by_id, seq_ids, new_task, config) -> bool:
    finish, travel = route_finish_and_travel(robot, tasks_by_id, seq_ids, new_task)
    return finish <= new_task.deadline and travel * config.battery_cost_per_unit_dist <= robot.battery + 1e-9
