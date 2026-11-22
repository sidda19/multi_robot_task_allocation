"""Consistency audit for the final per-seed experiment outputs."""
import json
import os
import sys
import numpy as np
import pandas as pd


def run_math_audit():
    raw_path = "results/experiment_results.csv"
    agg_path = "results/aggregated_results.csv"
    config_path = "results/experiment_config.json"
    assert os.path.isfile(raw_path), f"Missing {raw_path}"
    assert os.path.isfile(agg_path), f"Missing {agg_path}"
    assert os.path.isfile(config_path), f"Missing {config_path}"

    raw = pd.read_csv(raw_path)
    agg = pd.read_csv(agg_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert len(raw) == 380, f"Expected 380 raw runs, found {len(raw)}"
    assert sorted(raw["seed"].unique().tolist()) == list(range(1, 11))
    expected_counts = {
        "Exact ILP (assignment)": 10,
        "Exact ILP (sequence-aware)": 10,
        "Centralized Greedy": 10,
        "Distributed Greedy": 50,
        "SSI Auction": 50,
        "Cooperative MARL (IDQN)": 50,
        "Cooperative MARL (shared DQfD)": 50,
        "Online Distributed Greedy": 50,
        "Online Greedy (no-message convention)": 50,
        "Random Masked Policy": 50
    }
    assert raw["method"].value_counts().to_dict() == expected_counts
    print("[PASS] 380 rows, 10 seeds, and per-method run counts.")

    # Verify each static gap against the same-seed ILP objective.
    ilp = raw[raw.method == "Exact ILP (assignment)"].set_index("seed")
    static_methods = ["Centralized Greedy", "Distributed Greedy", "SSI Auction"]
    for _, row in raw[raw.method.isin(static_methods)].iterrows():
        ref = float(ilp.loc[row.seed, "assignment_objective_cost"])
        obj = float(row["assignment_objective_cost"])
        expected_gap = (obj - ref) / ref * 100.0
        actual_gap = float(row["optimality_gap_%"])
        assert np.isclose(actual_gap, expected_gap, atol=0.02), (
            f"Gap mismatch for {row.method}/{row.comm_level}/seed={row.seed}: "
            f"{actual_gap} vs {expected_gap}"
        )
        assert actual_gap >= -1e-6, (
            f"Static method objective beats the purported ILP optimum: "
            f"{row.method}/{row.comm_level}/seed={row.seed}, gap={actual_gap}"
        )
        allocation_unassigned = int(row["allocation_unassigned_count"])
        static_assignment_cost = float(row["static_assignment_cost"])
        expected_objective = static_assignment_cost + cfg["penalty_unassigned"] * allocation_unassigned
        assert np.isclose(obj, expected_objective, atol=0.02), (
            f"Static objective decomposition mismatch for {row.method}/{row.comm_level}/seed={row.seed}"
        )
    print("[PASS] Static objective decomposition and per-seed ILP gaps.")

    # Verify end-to-end decomposition and conflict accounting.
    penalty_u = float(cfg["penalty_unassigned"])
    penalty_c = float(cfg["penalty_conflict"])
    non_ilp = raw[raw.method != "Exact ILP (assignment)"]
    for _, row in non_ilp.iterrows():
        phys_cost = float(row["physical_execution_cost"])
        total_cost = float(row["total_cost"])
        alloc_penalty = float(row["allocation_conflict_penalty"])
        alloc_conflicts = int(row["allocation_conflicts"])
        physical_conflicts = int(row["physical_conflicts"])
        assert int(row["conflicts"]) == alloc_conflicts + physical_conflicts
        assert np.isclose(alloc_penalty, alloc_conflicts * penalty_c, atol=0.02)
        expected_total = phys_cost + alloc_penalty
        assert np.isclose(total_cost, expected_total, atol=0.02)
        physical_unassigned = int(row["physical_unassigned_count"])
        missed = int(row["missed_deadlines"])
        travel = float(row["travel_distance"])
        expected_physical = (
            travel + penalty_u * physical_unassigned
            + penalty_u * missed + penalty_c * physical_conflicts
        )
        assert np.isclose(phys_cost, expected_physical, atol=0.02), (
            f"Physical cost decomposition mismatch for {row.method}/{row.comm_level}/seed={row.seed}"
        )
        # Every task must be exactly one of: delivered on time, missed (late/unfinished), unassigned.
        completed = int(row["physically_completed_count"])
        assert completed + missed + physical_unassigned == cfg["n_tasks"], (
            f"Task partition broken for {row.method}/{row.comm_level}/seed={row.seed}: "
            f"{completed}+{missed}+{physical_unassigned} != {cfg['n_tasks']}"
        )
    print("[PASS] Physical cost, end-to-end cost, conflict decompositions and task partition (done+missed+unassigned = M).")

    # Sequence-aware exact reference: exact for the simulator and a lower bound for every method.
    seq = raw[raw.method == "Exact ILP (sequence-aware)"].set_index("seed")
    assert (pd.to_numeric(seq["missed_deadlines"]) == 0).all()
    for _, row in raw.iterrows():
        if row.method == "Exact ILP (assignment)":
            continue
        ref = float(seq.loc[row.seed, "total_cost"])
        gap = float(row["e2e_gap_vs_seq_opt_%"])
        assert np.isclose(gap, (float(row["total_cost"]) - ref) / ref * 100.0, atol=0.02)
        assert gap >= -1e-6, f"{row.method}/{row.comm_level}/seed={row.seed} beats the exact reference ({gap})"
    print("[PASS] Sequence-aware reference: zero missed deadlines and no method beats it.")

    # Verify runtime scope decomposition for every evaluated method.
    for _, row in raw.iterrows():
        if row.method == "Exact ILP (assignment)":
            assert np.isclose(float(row["runtime_ms"]), float(row["allocation_runtime_ms"]), atol=0.02)
            assert pd.isna(pd.to_numeric(pd.Series([row["physical_simulation_runtime_ms"]]), errors="coerce").iloc[0])
        elif row.method in ("Cooperative MARL (IDQN)", "Cooperative MARL (shared DQfD)", "Online Distributed Greedy", "Online Greedy (no-message convention)", "Random Masked Policy"):
            assert np.isclose(float(row["runtime_ms"]), float(row["physical_simulation_runtime_ms"]), atol=0.02)
            assert pd.isna(pd.to_numeric(pd.Series([row["allocation_runtime_ms"]]), errors="coerce").iloc[0])
        else:
            expected_runtime = float(row["allocation_runtime_ms"]) + float(row["physical_simulation_runtime_ms"])
            assert np.isclose(float(row["runtime_ms"]), expected_runtime, atol=0.02)
    print("[PASS] Runtime scope and decomposition.")

    # Verify communication budgets and L0 silence.
    levels = cfg["comm_levels"]
    for _, row in raw.iterrows():
        if row.comm_level == "N/A (Offline)" or row.method == "Centralized Greedy":
            continue
        level = row.comm_level
        budget = levels[level]["budget"]
        if budget != "unlimited":
            assert row["messages"] <= cfg["n_robots"] * float(budget) + 1e-9
        if level == "L0_None":
            assert row["messages"] == 0
    print("[PASS] Message budgets and L0 silence.")

    # Recompute all available aggregate numeric means and sample standard deviations.
    check_cols = [
        "total_cost", "physical_execution_cost", "allocation_conflict_penalty",
        "assignment_cost", "assignment_objective_cost", "static_assignment_cost",
        "optimality_gap_%", "assignment_rate_%", "physical_completion_rate_%",
        "conflicts", "allocation_conflicts", "physical_conflicts", "messages",
        "travel_distance", "runtime_ms", "allocation_runtime_ms",
        "physical_simulation_runtime_ms", "e2e_gap_vs_seq_opt_%", "missed_deadlines",
        "physically_completed_count"
    ]
    for _, arow in agg.iterrows():
        subset = raw[(raw.method == arow.method) & (raw.comm_level == arow.comm_level)]
        for col in check_cols:
            values = pd.to_numeric(subset[col], errors="coerce").dropna()
            mean_col, std_col = f"{col}_mean", f"{col}_std"
            agg_mean = pd.to_numeric(pd.Series([arow.get(mean_col)]), errors="coerce").iloc[0]
            agg_std = pd.to_numeric(pd.Series([arow.get(std_col)]), errors="coerce").iloc[0]
            if len(values) == 0:
                assert pd.isna(agg_mean) and pd.isna(agg_std), (
                    f"Expected N/A aggregate for {arow.method}/{arow.comm_level}/{col}"
                )
            else:
                assert np.isclose(values.mean(), agg_mean, atol=0.01)
                expected_std = values.std(ddof=1) if len(values) > 1 else 0.0
                assert np.isclose(expected_std, agg_std, atol=0.01)
    print("[PASS] Aggregated means and sample standard deviations match raw data.")

    required_figures = [
        "fig1_total_cost_vs_comm.png", "fig2_optimality_gap_vs_comm.png",
        "fig3_completion_rate_vs_comm.png", "fig4_conflicts_vs_comm.png",
        "fig5_message_overhead_vs_comm.png", "fig6_runtime_vs_comm.png",
        "fig7_gap_to_exact_reference_vs_comm.png"
    ]
    for name in required_figures:
        path = os.path.join("results", "figures", name)
        assert os.path.isfile(path) and os.path.getsize(path) > 1000, f"Missing/empty plot: {path}"
    assert os.path.isfile("results/summary_tables.md")
    print("[PASS] Seven figures and summary table exist.")
    print("ALL MATHEMATICAL AND OUTPUT CONSISTENCY CHECKS PASSED.")


if __name__ == "__main__":
    run_math_audit()
