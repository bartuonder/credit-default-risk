# Decisions

Format: date, decision, reason. Newest entries at the bottom of each section.

## Decisions

- 2026-11-16: Work happens on the branch assigned to this repository. Reason: pushes to other branch names are not permitted in this environment.
- 2026-11-16: The configured git author identity is left unchanged (rule: do not change it). The identity name matches one of the forbidden search terms, so a search over `git log` authors will report it; this cannot be fixed without changing the identity. Reason: the instruction not to change the identity takes precedence.
- 2026-11-16: The `data/raw/` file is kept tracked even though it is 5.5 MB. Reason: it was supplied in the repository and is the single source of truth for the pipeline.
- 2026-11-16: Excel is read with the `calamine` engine of pandas. Reason: the `xlrd` engine is deprecated in the installed pandas version.
- 2026-11-16: Dependencies are pinned to the versions installed in the working environment (Python 3.13). Reason: reproducibility.
- 2026-11-16: The dataset page (archive.ics.uci.edu) is not reachable from the sandbox network, so license and citation in `docs/data.md` are recorded from established knowledge of the page (CC BY 4.0, Yeh and Lien 2009) and marked as not re-verified. Reason: network limits.
- 2026-11-16: The hold-out split is created before EDA, and EDA uses the training split only. Reason: the test set must stay untouched until the final evaluation; `credit_risk.split.load_test` is called only by the final evaluation stage.
- 2026-11-16: `E501` is ignored by the linter; the formatter wraps code and long report strings are left as they are. Reason: f-string report lines cannot be wrapped automatically.
- 2026-11-16: The download fallback script could not be exercised (OpenML and UCI are blocked in the build network); it is not needed because the raw file is present.

## Open issues

None yet.
