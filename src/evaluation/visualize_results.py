"""Generate six plots and a data-derived Markdown results summary."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


COMM_ORDER = ["L0_None", "L1_Low", "L2_Medium", "L3_High", "L4_Full"]
COMM_LABELS = ["L0\n(None)", "L1\n(Low)", "L2\n(Med)", "L3\n(High)", "L4\n(Full)"]
COMM_METHODS = [
    "Distributed Greedy", "SSI Auction", "Online Distributed Greedy", "Online Greedy (no-message convention)",
    "Cooperative MARL (IDQN)", "Cooperative MARL (shared DQfD)", "Random Masked Policy"
]


def _num(row, key):
    return pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]


def _fmt(mean, std, suffix=""):
    if pd.isna(mean):
        return "N/A"
    if pd.isna(std):
        return f"{mean:.2f}{suffix}"
    return f"{mean:.2f} ± {std:.2f}{suffix}"


def generate_visualizations(
    agg_csv_path="results/aggregated_results.csv",
    raw_csv_path="results/experiment_results.csv",
    output_dir="results/figures",
    summary_md_path="results/summary_tables.md"
):
    print("=" * 80)
    print("Generating corrected publication figures and data-derived summary tables...")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    agg = pd.read_csv(agg_csv_path)
    raw = pd.read_csv(raw_csv_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--"
    })
    styles = {
        "Centralized Greedy": dict(color="#666666", marker="o", linestyle=":", label="Centralized Greedy"),
        "Distributed Greedy": dict(color="#1f77b4", marker="o", linestyle="-", label="Distributed Greedy (static plan)"),
        "Online Distributed Greedy": dict(color="#9467bd", marker="s", linestyle="-", label="Online Distributed Greedy"),
        "Online Greedy (no-message convention)": dict(color="#e377c2", marker="P", linestyle="-", label="Online Greedy (convention)"),
        "Random Masked Policy": dict(color="#8c564b", marker="x", linestyle="--", label="Random Masked Policy"),
        "SSI Auction": dict(color="#ff7f0e", marker="^", linestyle="-", label="SSI Auction"),
        "Cooperative MARL (shared DQfD)": dict(color="#17becf", marker="v", linestyle="-", label="Cooperative MARL (shared DQfD)"),
        "Cooperative MARL (IDQN)": dict(color="#2ca02c", marker="D", linestyle="-", label="Cooperative MARL (IDQN)")
    }
    x = np.arange(len(COMM_ORDER))
    cg = agg[agg.method == "Centralized Greedy"].iloc[0]

    def series(method, mean_col, std_col):
        sub = agg[agg.method == method].set_index("comm_level").reindex(COMM_ORDER)
        mean = pd.to_numeric(sub[mean_col], errors="coerce").to_numpy(dtype=float)
        std = pd.to_numeric(sub[std_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        return mean, std

    def save_plot(filename, ylabel, title, value_col, std_col, baseline_col=None, log=False, methods=COMM_METHODS):
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=250)
        if baseline_col:
            baseline = _num(cg, baseline_col)
            if pd.notna(baseline):
                ax.axhline(
                    baseline, color=styles["Centralized Greedy"]["color"],
                    linestyle=":", linewidth=2,
                    label=f"Centralized Greedy ({baseline:.1f})"
                )
        for method in methods:
            mean, std = series(method, value_col, std_col)
            st = styles[method]
            ax.errorbar(x, mean, yerr=std, capsize=4, label=st["label"],
                        color=st["color"], marker=st["marker"], linestyle=st["linestyle"])
        ax.set_xticks(x)
        ax.set_xticklabels(COMM_LABELS)
        ax.set_xlabel("Communication Level")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=250)
        plt.close(fig)
        print(f"Saved: {path}")

    # End-to-end includes physical execution cost plus allocation-stage conflict penalty.
    save_plot(
        "fig1_total_cost_vs_comm.png",
        "End-to-end total cost (lower is better)",
        "End-to-End System Cost vs Communication Level (Mean ± SD, 10 Seeds)",
        "total_cost_mean", "total_cost_std",
        baseline_col="total_cost_mean"
    )
    save_plot(
        "fig2_optimality_gap_vs_comm.png",
        "Static-objective gap vs assignment-model ILP (%)",
        "Static Objective Gap to the (relaxed) Assignment ILP vs Communication Level",
        "optimality_gap_%_mean", "optimality_gap_%_std",
        baseline_col="optimality_gap_%_mean",
        methods=["Distributed Greedy", "SSI Auction"]
    )
    save_plot(
        "fig3_completion_rate_vs_comm.png",
        "On-time physical completion rate (%)",
        "On-Time Physical Completion vs Communication Level",
        "physical_completion_rate_%_mean", "physical_completion_rate_%_std"
    )
    # Conflict count is allocation conflicts + physical conflicts for static
    # allocators; MARL conflicts occur in its online episode.
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=250)
    width = 0.8 / len(COMM_METHODS)
    for idx, method in enumerate(COMM_METHODS):
        mean, std = series(method, "conflicts_mean", "conflicts_std")
        ax.bar(x + (idx - (len(COMM_METHODS) - 1) / 2) * width, mean, yerr=std, width=width,
               label=styles[method]["label"], color=styles[method]["color"],
               capsize=4, alpha=0.85, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(COMM_LABELS)
    ax.set_xlabel("Communication Level")
    ax.set_ylabel("Conflict count per episode")
    ax.set_title("Conflict Count vs Communication Level (Mean ± SD, 10 Seeds)")
    ax.legend(framealpha=0.9); fig.tight_layout()
    path = os.path.join(output_dir, "fig4_conflicts_vs_comm.png")
    fig.savefig(path, dpi=250); plt.close(fig); print(f"Saved: {path}")

    save_plot(
        "fig5_message_overhead_vs_comm.png",
        "Transmitted messages per episode",
        "Communication Message Overhead vs Communication Level",
        "messages_mean", "messages_std"
    )
    save_plot(
        "fig6_runtime_vs_comm.png",
        "Total evaluation runtime per episode (ms, log scale)",
        "Comparable End-to-End Evaluation Runtime vs Communication Level",
        "runtime_ms_mean", "runtime_ms_std", log=True
    )

    save_plot(
        "fig7_gap_to_exact_reference_vs_comm.png",
        "End-to-end cost gap to sequence-aware exact optimum (%)",
        "Gap to the Exact Physical Reference vs Communication Level",
        "e2e_gap_vs_seq_opt_%_mean", "e2e_gap_vs_seq_opt_%_std",
        baseline_col="e2e_gap_vs_seq_opt_%_mean"
    )

    # Summary is generated exclusively from the current aggregate CSV.
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Experimental Evaluation Summary\n\n")
        f.write("## Multi-Robot Task Allocation Under Limited Communication (MRTA-LC)\n\n")
        f.write("**Course Project — Group 45**\n\n")
        f.write(f"Raw observations: **{len(raw)} runs** across **{raw.seed.nunique()} seeds**. "
                "All displayed values are mean ± sample standard deviation unless noted.\n\n")
        f.write("### Metric definitions\n\n")
        f.write("- **End-to-end total cost:** physical execution cost + allocation-stage conflict penalty. "
                "This is the `total_cost` column.\n")
        f.write("- **Physical execution cost:** travel distance + penalties for unassigned tasks, missed deadlines, "
                "and conflicts observed during physical execution.\n")
        f.write("- **Static assignment objective:** estimated assignment cost + unassigned-task penalty. "
                "It excludes conflict overhead and is the only objective compared with Exact ILP.\n")
        f.write("- **ILP gap (static):** static assignment objective relative to the assignment-model Exact ILP. "
                "Defined only for the static allocators; it is an optimistic reference because it ignores "
                "that a robot serves its tasks one after another.\n")
        f.write("- **Exact physical reference (sequence-aware ILP):** the exact optimum of the simulator's "
                "physical cost (travel + penalty for unassigned tasks); replaying its plan gives zero missed deadlines.\n")
        f.write("- **Gap to exact physical reference:** (end-to-end cost - reference) / reference, defined for every "
                "method that is physically executed, including the online policies.\n")
        f.write("- **Runtime:** static methods include allocator + physical replay; MARL includes the online "
                "policy and physical episode. Exact ILP reports solver time only and is not plotted.\n\n")

        f.write("### Table 1. Results by method and communication level\n\n")
        f.write("| Method | Comm | End-to-end cost | Gap to exact physical ref. | Static objective | Static ILP gap | Assignment | On-time completion | Conflicts | Messages | Runtime ms |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in agg.iterrows():
            f.write(
                f"| {row['method']} | {row['comm_level']} | "
                f"{_fmt(_num(row,'total_cost_mean'),_num(row,'total_cost_std'))} | "
                f"{_fmt(_num(row,'e2e_gap_vs_seq_opt_%_mean'),_num(row,'e2e_gap_vs_seq_opt_%_std'),'%')} | "
                f"{_fmt(_num(row,'assignment_objective_cost_mean'),_num(row,'assignment_objective_cost_std'))} | "
                f"{_fmt(_num(row,'optimality_gap_%_mean'),_num(row,'optimality_gap_%_std'),'%')} | "
                f"{_fmt(_num(row,'assignment_rate_%_mean'),_num(row,'assignment_rate_%_std'),'%')} | "
                f"{_fmt(_num(row,'physical_completion_rate_%_mean'),_num(row,'physical_completion_rate_%_std'),'%')} | "
                f"{_fmt(_num(row,'conflicts_mean'),_num(row,'conflicts_std'))} | "
                f"{_fmt(_num(row,'messages_mean'),_num(row,'messages_std'))} | "
                f"{_fmt(_num(row,'runtime_ms_mean'),_num(row,'runtime_ms_std'))} |\n"
            )

        f.write("\n### Table 2. Pre-stated hypotheses checked against the data (descriptive)\n\n")
        f.write("Thresholds are the ones stated before the experiments. Paired per-seed tests are produced "
                "separately by `scripts/paired_tests.py` (`results/paired_tests.md`).\n\n")
        f.write("| Hypothesis | Method | Observation | Verdict by stated criterion |\n|---|---|---|---|\n")

        def val(method, comm, col):
            hit = agg[(agg.method == method) & (agg.comm_level == comm)]
            return _num(hit.iloc[0], col) if not hit.empty else float("nan")

        for method in ["Online Distributed Greedy", "Distributed Greedy", "Cooperative MARL (shared DQfD)"]:
            costs = [val(method, c, "total_cost_mean") for c in COMM_ORDER[:4]]
            if all(pd.notna(c) for c in costs):
                monotone = all(costs[i] >= costs[i + 1] - 1e-9 for i in range(3))
                lower = costs[3] < costs[0]
                verdict = ("supported" if (lower and monotone) else
                           "partly supported (L3 below L0 but not monotone)" if lower else "not supported")
                f.write(f"| H1: cost falls as communication rises (L0 to L3) | {method} | "
                        f"end-to-end cost {' -> '.join(f'{c:.1f}' for c in costs)} | {verdict} |\n")
        for method in ["Online Distributed Greedy", "Distributed Greedy", "Cooperative MARL (shared DQfD)"]:
            c0 = val(method, "L0_None", "conflicts_mean")
            c2 = val(method, "L2_Medium", "conflicts_mean")
            if pd.notna(c0) and c0 > 0 and pd.notna(c2):
                red = (c0 - c2) / c0 * 100
                f.write(f"| H2: conflicts fall by more than 80% from L0 to L2 | {method} | "
                        f"{c0:.2f} -> {c2:.2f} ({red:.1f}% reduction) | "
                        f"{'supported' if red > 80 else 'not supported'} |\n")
        for comm in ["L1_Low", "L2_Medium"]:
            for marl_name in ["Cooperative MARL (IDQN)", "Cooperative MARL (shared DQfD)"]:
                names = [marl_name, "Online Distributed Greedy", "Distributed Greedy", "Random Masked Policy"]
                row = {m: (val(m, comm, "conflicts_mean"), val(m, comm, "physical_completion_rate_%_mean"),
                           val(m, comm, "total_cost_mean")) for m in names}
                text = "; ".join(f"{m.split(' (')[0] if m != marl_name else m}: {c:.2f} conflicts, {p:.0f}% on time, cost {k:.0f}"
                                 for m, (c, p, k) in row.items() if pd.notna(c))
                marl_c, marl_p, marl_k = row[marl_name]
                rnd_k = row["Random Masked Policy"][2]
                g_k = row["Online Distributed Greedy"][2]
                d_k = row["Distributed Greedy"][2]
                learned = marl_k < 0.95 * rnd_k            # 'meaningfully' cheaper than random: at least 5% lower
                verdict = (("more than 5% cheaper than a random masked policy (check the paired tests for significance)" if learned
                            else "NOT meaningfully (>5%) cheaper than a random masked policy: learning is not demonstrated")
                           + "; " + ("cheaper than" if marl_k < 0.95 * g_k else "not clearly cheaper than") + " online greedy"
                           + "; " + ("cheaper than" if marl_k < 0.95 * d_k else "not clearly cheaper than") + " the static Distributed Greedy plan")
                f.write(f"| H3: MARL coordinates under sparse communication ({comm}) | {marl_name} | {text} | {verdict} |\n")

        f.write("\n### Table 3. Communication trade-off\n\n")
        f.write("| Method | Comm | Messages | End-to-end cost | Change in cost vs L0 | Static objective gap to assignment ILP |\n|---|---|---:|---:|---:|---:|\n")
        for method in COMM_METHODS:
            sub = agg[agg.method == method].set_index("comm_level").reindex(COMM_ORDER)
            base_cost = _num(sub.loc["L0_None"],"total_cost_mean")
            for comm in COMM_ORDER:
                row = sub.loc[comm]
                messages = _num(row,"messages_mean")
                cost = _num(row,"total_cost_mean")
                gap = _num(row,"optimality_gap_%_mean")
                delta = cost-base_cost if pd.notna(base_cost) and pd.notna(cost) else float("nan")
                gap_text = f"{gap:.1f}%" if pd.notna(gap) else "N/A (online policy)"
                delta_text = f"{delta:+.1f}" if pd.notna(delta) else "N/A"
                f.write(f"| {method} | {comm} | {messages:.1f} | {cost:.1f} | {delta_text} | {gap_text} |\n")

    print(f"Summary table successfully written to: {summary_md_path}")


if __name__ == "__main__":
    generate_visualizations()
