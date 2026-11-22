"""Writes results/report_table.md: the compact results table used in the report (from the raw CSV)."""
import pandas as pd

ORDER = ["Exact ILP (assignment)", "Exact ILP (sequence-aware)", "Centralized Greedy", "Distributed Greedy", "SSI Auction",
         "Online Distributed Greedy", "Online Greedy (no-message convention)", "Cooperative MARL (shared DQfD)", "Cooperative MARL (IDQN)", "Random Masked Policy"]
raw = pd.read_csv("results/experiment_results.csv", keep_default_na=False, na_values=["N/A"])
for c in ["end_to_end_total_cost", "physical_execution_cost", "physically_completed_count", "conflicts", "messages", "e2e_gap_vs_seq_opt_%"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
g = raw.groupby(["method", "comm_level"], sort=False).mean(numeric_only=True)
rep = pd.read_csv("results/ilp_replay.md", sep="|", skiprows=2, skipinitialspace=True).dropna(axis=1, how="all")
rep.columns = [c.strip() for c in rep.columns]
rep["method"] = rep["method"].str.strip()
rep = rep.set_index("method")
ilp_cost = float(rep.loc["Exact ILP (assignment)", "physical_cost"]); ilp_ot = float(rep.loc["Exact ILP (assignment)", "on_time_tasks"])
ref_cost = float(rep.loc["Exact ILP (sequence-aware)", "physical_cost"])
lines = ["| Method | Level | End-to-end cost | Gap to exact ref. (%) | On-time tasks (of 10) | Conflicts | Messages |", "|----------------------------------|-----|--------:|--------:|--------:|--------:|--------:|"]
for m in ORDER:
    for (mm, lvl), r in g.iterrows():
        if mm != m:
            continue
        lv = lvl.split("_")[0] if lvl.startswith("L") else ("Global" if lvl.startswith("Global") else "Offline")
        if m == "Exact ILP (assignment)":
            lines.append(f"| Exact ILP (assignment), replayed | {lv} | {ilp_cost:.1f}* | {(ilp_cost-ref_cost)/ref_cost*100:.1f}* | {ilp_ot:.1f}* | 0.0 | 0.0 |"); continue
        lines.append(f"| {m.replace('Online Greedy (no-message convention)','Online Greedy (convention)').replace('Cooperative MARL (shared DQfD)','MARL shared DQfD (new)').replace('Cooperative MARL (IDQN)','MARL IDQN (original)')} | {lv} | {r.end_to_end_total_cost:.1f} | {r['e2e_gap_vs_seq_opt_%']:.1f} | {r.physically_completed_count:.1f} | {r.conflicts:.1f} | {r.messages:.1f} |")
open("results/report_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
