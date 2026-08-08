# P6 — Model pilot on CA_1

**Output:** `notebooks/06_model_pilot.ipynb`
**Effort:** 2–3 sessions
**Gate:** at least one model clearly beats seasonal-naive; runtimes measured and extrapolated to 10 stores

## Objective

Build and validate the **entire modelling pipeline end-to-end on one store** (`CA_1`, ~3,049 series),
where iteration takes minutes rather than hours. P7 is a rerun of this notebook at wider scope, so
every design question should be settled here, before it costs ten stores' worth of compute.

## Assumes

- P3 passed: `src/metrics.py` validated.
- P4 passed: seasonal-naive WRMSSE recorded as the bar.
- P5 passed: `src/features.py` importable, CA_1 partition built.

## Validation design

Rolling-origin, three folds of 28 days, each trained only on days before its fold:

| Fold | Trains on | Predicts |
|---|---|---|
| 1 | `d_1` – `d_1829` | `d_1830` – `d_1857` |
| 2 | `d_1` – `d_1857` | `d_1858` – `d_1885` |
| 3 | `d_1` – `d_1885` | `d_1886` – `d_1913` |

Identical to P4's folds, so baselines and models are directly comparable.

## The four candidates

| Model | Shape | Configuration | Why it's here |
|---|---|---|---|
| **LightGBM Tweedie** | global, per store | `objective=tweedie`, `tweedie_variance_power=1.1`, `learning_rate=0.03`, `num_leaves=128`, `min_data_in_leaf=100`, `feature_fraction=0.6`, `bagging_fraction=0.8`, early stopping | Tweedie handles the zero-inflated, positively-skewed count distribution directly. This family won M5. |
| **LightGBM Poisson** | global, per store | `objective=poisson`, otherwise matched | Same features, different loss — isolates the objective's contribution, and the two blend well. |
| **XGBoost** | global, per store | `tree_method=hist`, `objective=count:poisson`, `max_depth=8`, `eta=0.03`, `subsample=0.8`, `colsample_bytree=0.6`, `min_child_weight=50` | Different split algorithm and regularisation on identical features — a real check that results aren't a LightGBM artifact. |
| **Nixtla StatsForecast** | **per series** | `CrostonOptimized`, `ADIDA`, `IMAPA`, `TSB(0.2, 0.2)`, `SeasonalNaive(7)`, `DynamicOptimizedTheta(7)`, `n_jobs=-1` | Purpose-built for intermittent demand, no feature engineering, and the only per-series arm. The contrast is informative in itself. |

On model *shape*: the three gradient-boosted models are **global** — one model learns across all
series in the store, with `item_id`/`dept_id`/`cat_id` as categoricals, so cross-series structure is
learned rather than discarded. Do not fit one model per series; each series has ≤1,941 observations,
~68% of them zero, and per-series fitting throws away exactly the transferable patterns (SNAP, events,
price response) that matter.

## Tuning

Keep it light: sane defaults, early stopping on the fold, and a small sweep on `learning_rate` and
`num_leaves`. On M5, features and validation design move the score far more than hyperparameters do,
and an exhaustive search on 8 cores is not a good use of the time.

## Runtime measurement — a gate item

Record wall-clock training time per model per fold. Extrapolate to 10 stores and put the estimate in
`DECISIONS.md`. Rough expectation: 10–20 min per store per gradient-boosted config, so 2–4 hours for
a full 10-store run. If the pilot suggests materially worse, resolve it here — the fallbacks are a
reduced history window, store×category partitions, or fewer boosting rounds.

## Optional: alternative horizon strategy

If the single-model lag-28 approach underperforms, the main alternative used by strong M5 solutions
is to **split the horizon into four 7-day blocks and fit a separate model per block.** Days 1–7 can
then use lags as short as 7, which is genuinely more information than lag-28 allows. Cost: 4× the
models and 4× the training time.

Test it in the pilot precisely because it is cheap here. Decide on evidence, and record the decision.

## Gate

- [ ] All four candidates scored on all three folds with `src/metrics.py`
- [ ] At least one clearly beats the P4 seasonal-naive WRMSSE
- [ ] Per-model runtime measured and extrapolated to 10 stores
- [ ] Horizon strategy (single vs 4×7-day) decided and recorded
- [ ] Per-level and per-ADI/CV²-quadrant breakdown produced for the best model

## Invariants

- Folds end at `d_1913`; **the holdout is not touched in this phase** (invariant 1).
- Features come from `src/features.py`, not redefined in the notebook (see P5).
- Chronological splits only (invariant 5).

Full list: [README](README.md#invariants).
