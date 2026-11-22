"""The former report generator used hard-coded experimental values.

It has been retired to prevent accidentally overwriting the report with stale
results. Build the final course report from the current results/summary_tables.md
and report/REPORT_UPDATE_REQUIRED.md, then export and visually verify PDF/DOCX.
"""

if __name__ == "__main__":
    raise SystemExit(
        "Report generation is disabled: the previous script embedded stale hard-coded results. "
        "Use the current results/summary_tables.md to update the final report before export."
    )
