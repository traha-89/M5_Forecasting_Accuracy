# M5 Forecasting Accuracy — project plan

Nine phases taking the raw M5 dataset through hygiene, EDA, a scored metric, baselines, features,
four candidate models, and a final 28-day forecast.

**Working a single phase? Read only that phase's brief.** Each is self-contained — it states what it
can assume already exists, what to build, and the gate it must pass. Reading the whole plan to build
one phase wastes context and is not required.

A narrative version of this plan, with the reasoning behind each design decision, is at
[`plan.html`](plan.html) — open it in a browser.

## Horizon definitions

These underpin everything. `sales_train_evaluation.csv` carries `d_1`–`d_1941`; `calendar.csv` runs
28 days further, to `d_1969`.

| Region | Days | Dates | Actuals available? | Use |
|---|---|---|---|---|
| Training | `d_1`–`d_1913` | 2011-01-29 → 2016-04-24 | yes | Fit + rolling-origin CV. Everything through P6. |
| Holdout | `d_1914`–`d_1941` | 2016-04-25 → 2016-05-22 | **yes** | The honest test set. Scored exactly once, in P7. |
| Forecast | `d_1942`–`d_1969` | 2016-05-23 → 2016-06-19 | no | The final output. Refit on all 1,941 days, predict. |

The holdout actuals sit in the same CSV we train from, so peeking is trivially easy and silently
invalidates every reported number. See the invariants below.

`sample_submission.csv` has 60,980 rows: 30,490 `_validation` ids wanting `d_1914`–`d_1941`, then
30,490 `_evaluation` ids wanting `d_1942`–`d_1969`.

## Invariants

Repeated in every phase brief because they are load-bearing and easy to drop:

1. **Never train, tune, or select on `d_1914`–`d_1941`.** All model selection, hyperparameters, and
   ensemble weights are decided on rolling-origin folds ending at `d_1913` or earlier. The holdout is
   scored once, at the P7 gate.
2. **Every lag and rolling window is anchored at lag ≥ 28.** We predict 28 days in one shot, so on the
   last horizon day the most recent observed sale is 28 days old. Shorter lags are unavailable at
   prediction time.
3. **Per-store partitions only.** Never materialise all 10 stores of engineered features in one frame —
   that is ~14 GB against 15.9 GB of RAM. Sales `int16`, engineered features `float32`, ids `category`.
4. **Target encodings and aggregate statistics are fit on training folds only**, never on the full
   series including the validation window.
5. **No random train/test splits.** Time series only ever split chronologically.

## Phases

| Phase | Brief | Output | Gate |
|---|---|---|---|
| P0 | [Environment & scaffolding](P0-setup.md) | `requirements.txt`, dirs | All four model libraries import |
| P1 | [Load & hygiene check](P1-hygiene.md) | `01_data_hygiene.ipynb` → parquet | Assertion suite passes |
| P2 | [Exploratory analysis](P2-eda.md) | `02_eda.ipynb` + figures | Findings → feature-hypothesis table |
| P3 | [Metrics](P3-metrics.md) | `03_metrics.ipynb` + `src/metrics.py` | WRMSSE matches hand-computed toy panel |
| P4 | [Baselines](P4-baselines.md) | `04_baselines.ipynb` | Scored baseline table |
| P5 | [Feature engineering](P5-features.md) | `05_features.ipynb` + `src/features.py` | CA_1 builds <5 min, <8 GB peak |
| P6 | [Model pilot on CA_1](P6-pilot.md) | `06_model_pilot.ipynb` | A model clearly beats seasonal-naive |
| P7 | [Scale out & select](P7-scale.md) | `07_model_full.ipynb` + model card | Holdout WRMSSE reported once |
| P8 | [Final forecast](P8-forecast.md) | `08_final_forecast.ipynb` + submission | 60,980 rows, non-negative, plausible |

Phases run in order; each assumes its predecessors passed their gates.

## Recording decisions

At each gate, append what was decided and the number that justified it to
[`DECISIONS.md`](DECISIONS.md). This is what carries context between sessions — without it, later
phases re-derive choices inconsistently, and the pilot and full run end up with different feature
definitions.

## Workflow

One phase, one notebook, one PR against `main`, using `.github/PULL_REQUEST_TEMPLATE.md`.
Never push directly to `main`.

## Target machine

8 logical cores, 15.9 GB RAM, no CUDA GPU, Python 3.11 on Windows. Memory is the binding constraint
and is why invariant 3 exists.
