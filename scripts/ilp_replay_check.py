"""Replay the *assignment-model* Exact ILP plan in the physical simulator and
compare it with the sequence-aware exact optimum and Centralized Greedy.

Reproduces the numbers quoted in report Section 6 ("Assignment ILP vs reality").
Output: results/ilp_replay.md   Run from the project root: python scripts/ilp_replay_check.py
"""
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_CONFIG as cfg
from run_experiments import generate_scenario, execute_static_allocation_physically
from src.baselines.exact_ilp import ExactILPSolver
from src.baselines.exact_ilp_sequenced import SequenceAwareExactSolver
from src.baselines.centralized_greedy import CentralizedGreedyAllocator
from src.communication.channel import CommunicationChannel


def replay(plan, seed):
    return execute_static_allocation_physically(plan["assignments"], CommunicationChannel("L4_Full", cfg), seed, cfg)


rows = []
for seed in range(1, 11):
    robots, tasks = generate_scenario(seed)
    plans = {
        "Exact ILP (assignment)": ExactILPSolver(cfg).solve(copy.deepcopy(robots), copy.deepcopy(tasks)),
        "Exact ILP (sequence-aware)": SequenceAwareExactSolver(cfg).solve(copy.deepcopy(robots), copy.deepcopy(tasks)),
        "Centralized Greedy": CentralizedGreedyAllocator(cfg).solve(copy.deepcopy(robots), copy.deepcopy(tasks)),
    }
    for name, plan in plans.items():
        ph = replay(plan, seed)
        rows.append((seed, name, ph["total_physical_cost"], ph["physically_completed_count"], ph["missed_deadlines_count"]))
df = pd.DataFrame(rows, columns=["seed", "method", "physical_cost", "on_time_tasks", "missed_deadlines"])
m = df.groupby("method")[["physical_cost", "on_time_tasks", "missed_deadlines"]].mean().round(2)
out = ["# Physical replay of offline plans (10 seeds)", "", m.to_markdown(), ""]
Path("results/ilp_replay.md").write_text("\n".join(out))
print("\n".join(out))
