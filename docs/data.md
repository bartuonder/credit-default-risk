# Data card

## Source

- Dataset: Default of Credit Card Clients, UCI Machine Learning Repository (dataset id 350).
- Original study: Yeh, I. C., and Lien, C. H. (2009). The comparisons of data mining techniques for
  the predictive accuracy of probability of default of credit card clients. Expert Systems with
  Applications, 36(2), 2473-2480.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the
  dataset page. The page could not be re-checked from the build environment; verify it
  before redistributing.
- Scope: credit card clients of a bank in Taiwan, April to September 2005.

## Raw file

| Property | Value |
|---|---|
| File | `data/raw/default_of_credit_card_clients.xls` |
| SHA-256 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| Rows | 30,000 |
| Columns in the sheet | 25 (ID, 23 features, target) |
| Default rate (target = 1) | 22.12% (6,636 clients) |
| Exact duplicate rows (ignoring ID) | 56 (kept) |
| Rows with at least one negative bill amount | 1,930 (kept) |

The sheet has a descriptive row (X1 to X23) above the real header. `credit_risk.data` reads the
header from the second row and writes a clean CSV to `data/interim/default_clean.csv`.

## Columns

| Column | Type | Description |
|---|---|---|
| `LIMIT_BAL` | int | Credit limit in NT dollars (individual and family supplementary credit) |
| `SEX` | category | 1 male, 2 female |
| `EDUCATION` | category | 1 graduate school, 2 university, 3 high school, 4 other |
| `MARRIAGE` | category | 1 married, 2 single, 3 other |
| `AGE` | int | Age in years |
| `PAY_1` to `PAY_6` | int | Repayment status for September (`PAY_1`) back to April (`PAY_6`) 2005 |
| `BILL_AMT1` to `BILL_AMT6` | int | Bill statement amount, September back to April |
| `PAY_AMT1` to `PAY_AMT6` | int | Amount paid, September back to April |
| `default` | binary | 1 if the client defaulted in the following month (October 2005) |

Repayment status: -1 paid duly, 1 to 9 months of payment delay. The codes -2 and 0 are not
documented in the source; they are commonly read as "no consumption" and "revolving credit, minimum
payment made". They are kept as ordinal values and the model can treat them as it finds useful.

## Known quirks and the choices made

- The first repayment-status column is named `PAY_0` in the sheet while the others are `PAY_2` to
  `PAY_6`. It is renamed to `PAY_1`, so month indices line up with `BILL_AMT1` and `PAY_AMT1`.
- The target column `default payment next month` is renamed to `default`.
- `EDUCATION` contains the undocumented codes 0, 5 and 6 (345 rows in total). They are
  merged into category 4 ("other"). Raw counts per code are in `results/data_summary.json`.
- `MARRIAGE` contains the undocumented code 0 (54 rows). It is merged into category 3
  ("other").
- Amounts are skewed with a long right tail; the linear model uses a signed log transform, tree
  models use raw amounts.
- The data cover a single bank, one country and a six-month window around a local credit card
  crisis, so they do not represent current lending.

## Validation

`credit_risk.data.validate` checks column presence, nulls, allowed category codes, binary target,
positive limits, age range, repayment-status range and non-negative payments. It runs whenever the
clean CSV is built.
