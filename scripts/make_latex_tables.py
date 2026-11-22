"""Generate LaTeX result tables for the report directly from results/*.csv (no hand-typed numbers)."""
import pandas as pd, re, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
d = pd.read_csv(R/"results/experiment_results.csv", keep_default_na=False, na_values=["N/A"])
for c in ["end_to_end_total_cost","physically_completed_count","missed_deadlines","conflicts","messages",
          "assignment_objective_cost","assignment_rate_%","physical_execution_cost","allocation_conflict_penalty"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
LV = ["L0_None","L1_Low","L2_Medium","L3_High","L4_Full"]
def m(method, level, col):
    s = d[(d.method==method)&(d.comm_level==level)][col]
    return s.mean()
rows = [
 ("Original approaches", None),
 ("Centralized Greedy", "Centralized Greedy"),
 ("Distributed Greedy", "Distributed Greedy"),
 ("Cooperative MARL (IDQN)", "Cooperative MARL (IDQN)"),
 ("Teacher-requested baselines", None),
 ("SSI-style Sequential Auction", "SSI Auction"),
 ("Supplementary variants", None),
 ("Online Distributed Greedy", "Online Distributed Greedy"),
 ("Online Greedy, no-message convention", "Online Greedy (no-message convention)"),
 ("Cooperative MARL (shared DQfD)", "Cooperative MARL (shared DQfD)"),
 ("Random Masked Policy (control)", "Random Masked Policy"),
]
out = []
out.append(r"\begin{tabular}{@{}lcccccc@{}}")
out.append(r"\toprule")
out.append(r"Method & L0 & L1 & L2 & L3 & L4 & On-time (L2)\\")
out.append(r"\midrule")
for label, meth in rows:
    if meth is None:
        out.append(r"\multicolumn{7}{@{}l}{\emph{%s}}\\" % label); continue
    if meth == "Centralized Greedy":
        c = m(meth,"Global (Full Info)","end_to_end_total_cost"); k = m(meth,"Global (Full Info)","conflicts")
        o = m(meth,"Global (Full Info)","physically_completed_count")
        out.append(r"%s & \multicolumn{5}{c}{%.1f (%.1f), full information, no messages} & %.1f\\" % (label,c,k,o)); continue
    cells = []
    for lv in LV:
        cells.append("%.1f (%.1f)" % (m(meth,lv,"end_to_end_total_cost"), m(meth,lv,"conflicts")))
    out.append(r"%s & %s & %.1f\\" % (label, " & ".join(cells), m(meth,"L2_Medium","physically_completed_count")))
out.append(r"\bottomrule")
out.append(r"\end{tabular}")
(R/"report/table_main.tex").write_text("\n".join(out)+"\n")

# Table: assignment objective vs physical vs end-to-end (static planners)
rp = pd.read_csv(R/"results/ilp_replay.md", sep="|", skipinitialspace=True, skiprows=2).dropna(axis=1, how="all")
rp.columns = [c.strip() for c in rp.columns]
rp = rp.iloc[1:]; rp["method"] = rp["method"].str.strip()
for c in ["physical_cost","on_time_tasks","missed_deadlines"]: rp[c] = pd.to_numeric(rp[c])
def rep(name): return rp[rp.method==name].iloc[0]
def st(method, level):
    g = d[(d.method==method)&(d.comm_level==level)]
    return g
lines = []
lines.append(r"\begin{tabular}{@{}lrrrrrrr@{}}")
lines.append(r"\toprule")
lines.append(r"Plan & Assigned (\%) & Assign.\ obj. & Physical & Alloc.\ confl. & End-to-end & On-time & Missed\\")
lines.append(r"\midrule")
g = st("Exact ILP (assignment)","N/A (Offline)"); r_ = rep("Exact ILP (assignment)")
lines.append(r"Exact ILP, assignment model & %.0f & %.1f & %.1f$^{*}$ & 0.0 & %.1f$^{*}$ & %.1f$^{*}$ & %.1f$^{*}$\\" % (
    g["assignment_rate_%"].mean(), g.assignment_objective_cost.mean(), r_.physical_cost, r_.physical_cost, r_.on_time_tasks, r_.missed_deadlines))
g = st("Exact ILP (sequence-aware)","N/A (Offline)")
lines.append(r"Exact ILP, sequence-aware & %.0f & n/a & %.1f & 0.0 & %.1f & %.1f & %.1f\\" % (
    g["assignment_rate_%"].mean(), g.physical_execution_cost.mean(), g.end_to_end_total_cost.mean(), g.physically_completed_count.mean(), g.missed_deadlines.mean()))
for lab, meth, lv in [("Centralized Greedy","Centralized Greedy","Global (Full Info)"),("Distributed Greedy (L2)","Distributed Greedy","L2_Medium"),("SSI Auction (L2)","SSI Auction","L2_Medium")]:
    g = st(meth, lv)
    lines.append(r"%s & %.0f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f\\" % (lab, g["assignment_rate_%"].mean(), g.assignment_objective_cost.mean(), g.physical_execution_cost.mean(),
        g.allocation_conflict_penalty.mean(), g.end_to_end_total_cost.mean(), g.physically_completed_count.mean(), g.missed_deadlines.mean()))
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
(R/"report/table_metrics.tex").write_text("\n".join(lines)+"\n")
print("ok")
