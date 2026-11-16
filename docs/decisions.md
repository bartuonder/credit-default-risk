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
- 2026-11-16: The model is selected by the highest mean cross-validated PR-AUC; models whose paired difference to the top is not significant (corrected resampled t-test, alpha 0.05) are treated as ties and the smaller one is preferred (LightGBM over histogram gradient boosting over random forest). Reason: differences between tuned tree ensembles are within fold noise, and a small artifact is easier to ship.
- 2026-11-16: Calibration is applied only if it significantly lowers cross-validated log loss versus the raw model. Reason: raw probabilities were already well calibrated, and a calibrated ensemble would add size without measurable benefit.
- 2026-11-16: SEX and EDUCATION stay as model inputs, as in the original benchmark; the ablation reports what demographics contribute and the fairness diagnostic is descriptive. Reason: faithful reproduction of the benchmark; limits are documented.
- 2026-11-16: Fairness and error analysis are produced on out-of-fold training predictions and on the hold-out set; the hold-out figures are used in the report. Reason: the out-of-fold version needs no access to the test set.
- 2026-11-16: The `apt-get` step of the Dockerfile could not be run in the build sandbox (Debian mirrors are blocked by the egress policy). The remaining steps (pip install, non-root user, health check, serving the UI and API) were verified with a variant of the Dockerfile that borrows `libgomp` from the scikit-learn wheel. Reason: network limits.
- 2026-11-16: The 0.6 MB model artifact is committed; the image does not train at build time. Reason: artifact is far below 20 MB.

## Open issues

- Dockerfile `apt-get install libgomp1` step not executed in the sandbox; it runs in the CI Docker job.
- License and citation of the dataset page could not be re-checked from the sandbox (see decisions).
