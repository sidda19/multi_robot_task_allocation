"""Before/after table for the order-aware planners (priority-1 correction).

Runs Centralized Greedy, Distributed Greedy and the SSI auction twice on seeds 1-10:
  sequence_aware=False : legacy check (each task judged from t = 0, claims replayed in task-id order)
  sequence_aware=True  : deadline/battery checked against the robot's whole route (default)
and replays every plan in the same physical simulator.
Output: results/ablation_sequence_awareness.md     Run: python scripts/ablation_sequence_awareness.py
"""
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_CONFIG as cfg
from run_experiments import generate_scenario, execute_static_allocation_physically
from src.baselines.auction import AuctionAllocator
from src.baselines.centralized_greedy import CentralizedGreedyAllocator
from src.baselines.distributed_greedy import DistributedGreedyAllocator
from src.communication.channel import CommunicationChannel

rows = []
for seq_aware in (False, True):
    for seed in range(1, 11):
        robots, tasks = generate_scenario(seed)
        jobs = [("Centralized Greedy", "Global", CentralizedGreedyAllocator(cfg, sequence_aware=seq_aware), "L4_Full")]
        for lvl in ("L0_None", "L2_Medium", "L4_Full"):
            jobs.append(("Distributed Greedy", lvl, DistributedGreedyAllocator(CommunicationChannel(lvl, cfg), cfg, sequence_aware=seq_aware), lvl))
            jobs.append(("SSI Auction", lvl, AuctionAllocator(CommunicationChannel(lvl, cfg), cfg, sequence_aware=seq_aware), lvl))
        for name, lvl, alloc, replay_lvl in jobs:
            plan = alloc.solve(copy.deepcopy(robots), copy.deepcopy(tasks))
            ph = execute_static_allocation_physically(plan["assignments"], CommunicationChannel(replay_lvl, cfg), seed, cfg)
            rows.append(("order-aware" if seq_aware else "legacy", name, lvl,
                         ph["total_physical_cost"] + plan["conflict_penalty"],
                         ph["physically_completed_count"], ph["missed_deadlines_count"]))
df = pd.DataFrame(rows, columns=["variant", "method", "level", "end_to_end_cost", "on_time_tasks", "missed_deadlines"])
tab = df.groupby(["method", "level", "variant"])[["end_to_end_cost", "on_time_tasks", "missed_deadlines"]].mean().round(1)
out = "# Order-aware vs legacy static planners (10 seeds, same physical replay)\n\n" + tab.to_markdown() + "\n"
Path("results/ablation_sequence_awareness.md").write_text(out)
print(out)
