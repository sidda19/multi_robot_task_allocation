"""Preliminary comparison used to choose the online-greedy tie-break rule.

The rule was compared on seeds 11-50, which are NOT the evaluation seeds (1-10), so the
reported results are not tuned on the evaluation scenarios.
Output: results/tie_break_selection.md       Run: python scripts/tie_break_selection.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_CONFIG as cfg
from run_experiments import evaluate_marl_on_env
from src.baselines.online_policies import OnlineGreedyPolicy

LEVELS = ["L0_None", "L1_Low", "L2_Medium", "L3_High", "L4_Full"]
rows = []
for tie in ("none", "convention", "random"):
    for lvl in LEVELS:
        for seed in range(11, 51):
            r = evaluate_marl_on_env(OnlineGreedyPolicy(cfg, tie_break=tie, seed=1000 * seed + 7), lvl, seed, cfg)
            rows.append((tie, lvl, seed, r["total_cost"], r["physically_completed_count"], r["conflict_count"]))
df = pd.DataFrame(rows, columns=["tie_break", "level", "seed", "cost", "on_time", "conflicts"])
tab = df.groupby(["level", "tie_break"])[["cost", "on_time", "conflicts"]].mean().round(1)
out = "# Tie-break selection on seeds 11-50 (40 scenarios, not the evaluation seeds)\n\n" + tab.to_markdown() + "\n"
Path("results/tie_break_selection.md").write_text(out)
print(out)
