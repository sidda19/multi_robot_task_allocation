"""Paired per-seed tests used for the hypothesis section of the report.

Every comparison pairs the two conditions on the SAME seed (same warehouse
scenario). With only 10 seeds and one trained MARL checkpoint these tests are
exploratory: no multiple-comparison correction is applied and a significant
result says nothing about other trained checkpoints.
Output: results/paired_tests.md
"""
import numpy as np
import pandas as pd
from scipy import stats

RAW = "results/experiment_results.csv"
OUT = "results/paired_tests.md"


def _series(raw, method, comm, col):
    sub = raw[(raw.method == method) & (raw.comm_level == comm)].set_index("seed")[col]
    return pd.to_numeric(sub, errors="coerce").sort_index()


def _row(label, a, b, unit=""):
    diff = (a - b).dropna()
    if len(diff) < 2 or np.allclose(diff, 0):
        p_t = p_w = float("nan") if len(diff) < 2 else 1.0
    else:
        p_t = stats.ttest_rel(a.loc[diff.index], b.loc[diff.index]).pvalue
        p_w = stats.wilcoxon(a.loc[diff.index], b.loc[diff.index]).pvalue
    return (f"| {label} | {a.mean():.2f}{unit} | {b.mean():.2f}{unit} | {diff.mean():+.2f}{unit} | "
            f"{p_t:.4f} | {p_w:.4f} |")


def main():
    raw = pd.read_csv(RAW, keep_default_na=False, na_values=["N/A"])
    ODG, SDG, RND = "Online Distributed Greedy", "Distributed Greedy", "Random Masked Policy"
    MARLS = ("Cooperative MARL (shared DQfD)", "Cooperative MARL (IDQN)")
    cost, conf, done, gap = "total_cost", "conflicts", "physically_completed_count", "optimality_gap_%"
    lines = [
        "# Paired per-seed tests (exploratory)",
        "",
        "Pairs share the same seed. n = 10 seeds. p-values: paired t-test and Wilcoxon signed-rank (two-sided). "
        "No multiple-comparison correction. Each MARL result comes from ONE trained checkpoint.",
        "",
        "| Comparison (A vs B) | mean A | mean B | mean A-B | paired t p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
        "| **H1: cost vs communication** | | | | | |",
    ]
    for m in (ODG, SDG):
        for a, b in (("L0_None", "L1_Low"), ("L1_Low", "L2_Medium"), ("L2_Medium", "L3_High"), ("L0_None", "L3_High")):
            lines.append(_row(f"{m}: cost {a} vs {b}", _series(raw, m, a, cost), _series(raw, m, b, cost)))
    lines.append(_row(f"{SDG}: static ILP gap % L0 vs L3", _series(raw, SDG, "L0_None", gap), _series(raw, SDG, "L3_High", gap), "%"))
    lines.append("| **H2: conflicts vs communication** | | | | | |")
    for m in (ODG, SDG):
        lines.append(_row(f"{m}: conflicts L0 vs L2", _series(raw, m, "L0_None", conf), _series(raw, m, "L2_Medium", conf)))
    lines.append("| **H3: MARL vs references** | | | | | |")
    for MARL in MARLS:
        tag = "shared DQfD" if "shared" in MARL else "IDQN"
        for comm in ("L1_Low", "L2_Medium"):
            lines.append(_row(f"conflicts {comm}: MARL ({tag}) vs {SDG}", _series(raw, MARL, comm, conf), _series(raw, SDG, comm, conf)))
            lines.append(_row(f"conflicts {comm}: MARL ({tag}) vs {RND}", _series(raw, MARL, comm, conf), _series(raw, RND, comm, conf)))
            lines.append(_row(f"cost {comm}: MARL ({tag}) vs {RND}", _series(raw, MARL, comm, cost), _series(raw, RND, comm, cost)))
            lines.append(_row(f"cost {comm}: MARL ({tag}) vs {ODG}", _series(raw, MARL, comm, cost), _series(raw, ODG, comm, cost)))
            lines.append(_row(f"cost {comm}: MARL ({tag}) vs {SDG}", _series(raw, MARL, comm, cost), _series(raw, SDG, comm, cost)))
            lines.append(_row(f"on-time tasks {comm}: MARL ({tag}) vs {RND}", _series(raw, MARL, comm, done), _series(raw, RND, comm, done)))
            lines.append(_row(f"on-time tasks {comm}: MARL ({tag}) vs {ODG}", _series(raw, MARL, comm, done), _series(raw, ODG, comm, done)))
        lines.append(_row(f"cost L0_None: MARL ({tag}) vs {RND}", _series(raw, MARL, "L0_None", cost), _series(raw, RND, "L0_None", cost)))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
