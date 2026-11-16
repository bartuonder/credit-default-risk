# Credit Default Risk

Calibrated credit card default prediction on the UCI "Default of Credit Card Clients" data, with a repeated-CV model comparison, a cost-based decision threshold, SHAP explanations, and a FastAPI service with a small web interface.

> Educational demonstration on 2005 Taiwanese data. Not for real lending decisions.

## Results summary

Hold-out test set ({{n_test}} clients, evaluated once); intervals are 95% bootstrap intervals.

{{results_table}}

## Key findings

{{findings}}

## Quickstart

Requires Python 3.12 or newer. The raw data file is included in `data/raw/`.

```bash
make setup      # virtual environment and pinned dependencies
make train      # split, cross-validation, tuning, calibration, threshold, final fit
make evaluate   # one-time hold-out evaluation and model card
make run        # API and web interface on http://localhost:8000
```

Other targets: `make data` (Excel to clean CSV), `make eda`, `make report` (report and this README from `results/`), `make test`, `make lint`, `make all` (everything from scratch; stages are rerun with `FORCE=1`).

Docker:

```bash
docker build -t credit-risk:latest .
docker run --rm -p 8000:8000 credit-risk:latest
```

See `docs/deployment.md` for hosting and environment variables.

## Method

**Data.** {{n_rows}} clients, {{default_rate}} defaulted in the following month. Undocumented category codes (EDUCATION 0, 5, 6 and MARRIAGE 0) are merged into the documented "other" categories; the inconsistent column name `PAY_0` is renamed `PAY_1`. See `docs/data.md` for the data card, license and citation.

**Validation protocol.** A stratified 20% hold-out set ({{n_test}} clients) is created once with a fixed seed; its ids are stored in `results/split.json`, and the loader used for training and EDA cannot return it. All tuning and selection use the {{n_train}} training clients. Models are compared with repeated stratified {{n_splits}}-fold cross-validation ({{n_repeats}} repeats) on identical folds. Preprocessing (scaling, one-hot encoding, engineered features) sits inside the scikit-learn pipeline, so it is fitted on training folds only; tests check this.

**Features.** Raw columns plus engineered ones configured in `configs/config.yaml`: utilisation (bill / limit), payment-to-bill ratios, number and length of payment delays, a repayment-status trend and a bill trend.

**Models.** Majority-class dummy and logistic regression as baselines; random forest, histogram gradient boosting and LightGBM as candidates. Hyperparameters come from a random search (at most {{tuning_iter}} candidates, {{tuning_splits}}-fold CV on folds different from the evaluation folds). Models are compared with the corrected resampled t-test and the Wilcoxon signed-rank test on fold-wise differences. When the top candidates are not significantly different, the smaller model is chosen.

**Calibration.** Sigmoid and isotonic calibration (cross-validated, refitted inside each outer fold) are compared with the raw model on Brier score, log loss, expected calibration error and a Brier decomposition. A calibrator replaces the raw model only if the log-loss gain is significant.

**Cost-based decision.** A client is flagged when the predicted probability reaches a threshold chosen to minimise expected cost on out-of-fold predictions. Default assumption: a missed default costs {{cost_ratio}} times a declined good client. This is an assumption, not a measurement; the sensitivity analysis covers ratios 1 to 10.

**Interpretability.** Global and per-prediction SHAP values (tree explainer); one-hot columns are summed back to their feature. A feature-group ablation drops one group at a time and keeps one group at a time.

**Fairness diagnostic.** Approval rate, false positive rate and false negative rate by sex, age band and education, with Wilson intervals. It is descriptive and makes no claim of fairness.

## Results

### Model comparison (cross-validation on the training set)

{{comparison_table}}

Tuned candidates are statistically indistinguishable on PR-AUC (corrected t-test p-values {{pvalues}}); logistic regression is clearly behind (PR-AUC difference {{lr_gap}}).

![cross-validation scores](reports/figures/cv_comparison.png)

### Hold-out test result

{{test_table}}

### Calibration

{{calibration_table}}

![reliability diagrams](reports/figures/calibration_reliability.png)

### Cost curve

Chosen threshold {{threshold}} (cost ratio {{cost_ratio}}:1); the theoretical threshold for calibrated probabilities is {{bayes}}.

![threshold versus expected cost](reports/figures/cost_curve.png)

{{sensitivity_table}}

### SHAP

![SHAP summary](reports/figures/shap_summary.png)

![SHAP dependence](reports/figures/shap_dependence.png)

### Feature-group ablation

{{ablation_table}}

![ablation](reports/figures/ablation.png)

### Fairness diagnostic

{{fairness_text}}

![fairness diagnostic](reports/figures/fairness.png)

The complete report, with error analysis and all tables, is in `reports/report.md`; the exploratory analysis is in `reports/eda_summary.md`.

## API reference

Interactive documentation: `/docs`. Errors share one format: `{"error": {"code", "message", "details": [{"field", "message"}]}}`.

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness and whether the model is loaded |
| `GET /model-info` | Metrics, threshold, cost assumption, calibration curve, feature schema |
| `POST /predict` | Default probability, decision, risk band; `?explain=true&top_k=5` adds top factors |
| `POST /predict/batch` | Up to {{max_batch}} applicants per request; same query parameters |

Field names are the dataset columns in lower case: `limit_bal`, `sex` (1, 2), `education` (1 to 4), `marriage` (1 to 3), `age` (18 to 100), `pay_1` to `pay_6` (-2 to 9), `bill_amt1` to `bill_amt6`, `pay_amt1` to `pay_amt6` (0 or more). Month 1 is the most recent (September 2005). Unknown fields are rejected.

```bash
curl -s -X POST 'http://localhost:8000/predict?explain=true&top_k=3' \
  -H 'Content-Type: application/json' \
  -d '{{example_request}}'
```

Response (values produced by the trained model):

```json
{{example_response}}
```

Contributions are in the model's margin units (log-odds); positive values raise the predicted risk.

## Web interface

The interface is served at `/`: a form with example presets, a probability gauge, decision and risk band, a factor chart, and a Model tab with the test metrics, threshold, cost assumption and calibration curve. It follows the system light or dark setting.

![prediction view](docs/img/ui-predict-light.png)

![model view](docs/img/ui-model-light.png)

## Limitations

- The data are from one Taiwanese bank in 2005 and describe a specific credit-card cycle; they do not represent current customers or other markets.
- The 5:1 cost ratio is an assumption. Real costs depend on interest margins, collection costs and policy, and the sensitivity table shows how the threshold moves.
- There is no real-world validation: results are from a random hold-out split of the same period, with no temporal or external test.
- The fairness diagnostic is descriptive. Sex and education are model inputs here (as in the original benchmark); using them, or proxies, can be restricted in real lending. Some groups are small and their intervals wide.
- Cross-validated tuning on the training set is slightly optimistic for the selected model; the hold-out result is the unbiased estimate.
- SHAP values describe the model, not causal effects.

## Repository structure

```
configs/config.yaml        all settings (paths, features, CV, models, cost matrix)
data/raw/                  original Excel file (tracked); data/interim/ is generated
src/credit_risk/           data, features, models, cv, tuning, calibration, cost, explain, fairness, ...
src/credit_risk/api/       FastAPI service and static web interface
scripts/                   data card, report and README generators, download fallback
results/                   JSON and CSV outputs of every stage (source of all reported numbers)
artifacts/                 model.joblib and model_card.json
reports/                   EDA and evaluation report with figures
docs/                      data card, decisions log, deployment guide, screenshots
tests/                     unit and API tests (synthetic data only)
```

## Reproducibility

- Fixed seed ({{seed}}) in `configs/config.yaml`; fold seeds are derived from it. Dependencies are pinned in `requirements*.txt` and `pyproject.toml`.
- Every figure in this file and in `reports/report.md` is generated from `results/` or `artifacts/` by `scripts/make_report.py` and `scripts/make_readme.py`.
- The raw file SHA-256 is `{{sha}}` and is stored in `results/data_summary.json` and the model card.
- `artifacts/model_card.json` records metrics, threshold, cost assumption, feature schema, library versions and the git commit of the code that produced the model.
- Training uses the CPU only. A full `make all` takes roughly half an hour on four cores; stages that already have outputs are skipped by `make train`.

## License and citations

Code: MIT License (`LICENSE`). Data: CC BY 4.0 (UCI Machine Learning Repository).

Yeh, I. C., and Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

Dataset: Yeh, I. C. (2016). Default of Credit Card Clients. UCI Machine Learning Repository.
