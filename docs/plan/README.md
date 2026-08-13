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

**Why `d_1913`:** not a train/test ratio — `d_1941 − 28` and `d_1969 − 28 = d_1941`. Both blocks are
28 days (4 weeks) to match the real forecast horizon and avoid weekday/SNAP-cycle bias from an
off-week-boundary cut. `d_1914`–`d_1941` also matches Kaggle's original public-LB window.

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

## Working rules

- **If a gate fails, stop and report. Do not proceed to the next phase.** A failed assertion in P1
  that gets rationalised away propagates through six phases before anyone notices.
- **One phase, one notebook, one PR. Do not start the next phase in the same session.**
- **Do not invent file paths.** Everything that crosses a phase boundary is pinned in the data
  contract below. If something you need isn't there, that is a gap to raise, not to improvise.
- **Prefer failing loudly.** Assert expected shapes and counts rather than coercing or filling.

## Data contract

Every artifact that crosses a phase boundary. Produced by exactly one phase, consumed by the
phases listed. Do not deviate from these paths — later phases load them by name.

| Path | Produced | Consumed by | Contents |
|---|---|---|---|
| `data/processed/sales_long.parquet` | P1 | P2, P4, P5 | Long panel, pre-release rows dropped. `id` `category`, `d` `int16`, `sales` `int16`, `date` `datetime64`, `wm_yr_wk` `int16`, calendar columns, `item_id`/`dept_id`/`cat_id`/`store_id`/`state_id` `category`, `sell_price` `float32` |
| `data/processed/series_segments.parquet` | P2 | P6, P7 | One row per series: `id`, `adi` `float32`, `cv2` `float32`, `quadrant` `category` (smooth / erratic / intermittent / lumpy) |
| `src/metrics.py` | P3 | P4, P6, P7, P8 | `wrmsse()`, per-level breakdown, RMSSE, WMAE, MASE, bias |
| `reports/baselines.csv` | P4 | P6, P7 | `baseline`, `fold`, `wrmsse`, `rmsse`, `wmae`, `bias_pct` |
| `src/features.py` | P5 | P6, P7, P8 | `build_features(store_id, …)` — the single definition used by pilot and full run alike |
| `data/processed/features/store_id=<STORE>/part.parquet` | P5 | P6, P7, P8 | One partition per store. Engineered features `float32`, `sales` `int16`, ids `category` |
| `models/pilot/<model>_<fold>.<ext>` | P6 | — | Pilot models, CA_1 only |
| `reports/pilot_scores.csv` | P6 | P7 | `model`, `fold`, `wrmsse`, `rmsse`, `wmae`, `bias_pct`, `runtime_s` |
| `models/full/<model>_<store>.<ext>` | P7 | P8 | Ten models per configuration, persisted per store so a failure at store 8 doesn't lose 1–7 |
| `reports/full_scores.csv` | P7 | P8 | Same schema as `pilot_scores.csv`, plus `store_id` |
| `submission.csv` | P8 | — | 60,980 rows, `id` + `F1`–`F28` |
| `src/replenishment.py` | P9 (proposed) | — | Policy function(s) — (R,s,Q) reorder point + naive `(s,S)` baseline, same single-definition convention as `src/metrics.py`/`src/features.py` |
| `reports/replenishment_comparison.csv` | P9 (proposed) | — | `id`, `policy` (`rsq`/`naive`), `fold`, `stockout_rate`, `excess_stock`, `reorder_point` |

`data/` and `models/` are gitignored — these are build products, not committed artifacts.
`reports/*.csv` are small and **are** committed, so results survive between sessions.

Fold definitions, used identically by P4, P6, and P7:

| Fold | Trains on | Predicts |
|---|---|---|
| 1 | `d_1` – `d_1829` | `d_1830` – `d_1857` |
| 2 | `d_1` – `d_1857` | `d_1858` – `d_1885` |
| 3 | `d_1` – `d_1885` | `d_1886` – `d_1913` |

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

### Proposed (not yet approved)

| Phase | Brief | Output | Gate |
|---|---|---|---|
| P9 | [Replenishment policy](P9-replenishment.md) | `09_replenishment.ipynb` | Beats naive baseline on folds 1–3 |

Parked for future review, not part of the committed P0–P8 sequence above — see the brief for the
open dependency (P6/P7 produce point forecasts only; P9 needs a demand distribution) that must be
resolved before it can start.

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
