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

| Fold | Trains on | Predicts | Early-stops on |
|---|---|---|---|
| 1 | `d_1` – `d_1801` | `d_1830` – `d_1857` | `d_1802` – `d_1829` |
| 2 | `d_1` – `d_1829` | `d_1858` – `d_1885` | `d_1830` – `d_1857` |
| 3 | `d_1` – `d_1857` | `d_1886` – `d_1913` | `d_1858` – `d_1885` |

Same prediction windows as P4, so baselines and models are directly comparable. **Never early-stop on
the days being scored** — the stopping set is the 28 days immediately before the forecast window, and
training ends before that. Fold-local `training_end` and weight windows are in
[README](README.md#data-contract), same as P4.

Compare against the **`CA_1`-scope rows** of `reports/baselines.csv`. On a single store the 12
aggregation levels collapse to 4 distinct ones, so a pilot score is not comparable to a full-panel
bar.

## The four candidates

| Model | Shape | Configuration | Why it's here |
|---|---|---|---|
| **LightGBM Tweedie** | global, per store | `objective=tweedie`, `tweedie_variance_power` swept over 1.1 / 1.3 / 1.5, `learning_rate=0.03`, `num_leaves=128`, `min_data_in_leaf=100`, `feature_fraction=0.6`, `bagging_fraction=0.8`, early stopping | Tweedie handles the zero-inflated, positively-skewed count distribution directly. This family won M5. At `p=1.1` it is nearly Poisson, so the sweep is what makes the two arms distinct. |
| **LightGBM Poisson** | global, per store | `objective=poisson`, otherwise matched | Same features, different loss — isolates the objective's contribution, and the two blend well. |
| **XGBoost** | global, per store | `tree_method=hist`, `objective=count:poisson`, `max_depth=8`, `eta=0.03`, `subsample=0.8`, `colsample_bytree=0.6`, `min_child_weight=50` | Different split algorithm and regularisation on identical features — a real check that results aren't a LightGBM artifact. |
| **Nixtla StatsForecast** | **per series** | `CrostonOptimized`, `ADIDA`, `IMAPA`, `TSB(0.2, 0.2)`, `SeasonalNaive(7)`, `DynamicOptimizedTheta(7)`, `n_jobs=-1` — **scored individually, not as one blended arm** | Purpose-built for intermittent demand, no feature engineering, and the only per-series arm. The Croston family emits a flat 28-day forecast with no weekly shape, so a single blended number would hide which sub-model is driving the result. |

On model *shape*: the three gradient-boosted models are **global** — one model learns across all
series in the store, with `item_id`/`dept_id`/`cat_id` as categoricals, so cross-series structure is
learned rather than discarded. Do not fit one model per series; each series has ≤1,941 observations,
~68% of them zero, and per-series fitting throws away exactly the transferable patterns (SNAP, events,
price response) that matter.

## Tuning

Keep it light: sane defaults, early stopping on the inner split, and a small sweep on
`tweedie_variance_power` and `min_data_in_leaf` — at a 68.2% zero rate those are the parameters that
interact with the data's shape, where `learning_rate`/`num_leaves` mostly trade runtime for a
marginal gain. On M5, features and validation design move the score far more than hyperparameters
do, and an exhaustive search on 8 cores is not a good use of the time.

## Runtime measurement — a gate item

Record wall-clock training time per model per fold. Extrapolate to 10 stores and put the estimate in
`DECISIONS.md`. Rough expectation: 10–20 min per store per gradient-boosted config, so 2–4 hours for
a full 10-store run. If the pilot suggests materially worse, resolve it here — the fallbacks are a
reduced history window, store×category partitions, or fewer boosting rounds.

## Optional: alternative horizon strategy — skip by default

> **This is optional and not part of the gate. Do not attempt it until the four candidates above are
> built and scored.** Only pursue it if the best model fails to clearly beat seasonal-naive.

If the single-model lag-28 approach underperforms, the main alternative used by strong M5 solutions
is to **split the horizon into four 7-day blocks and fit a separate model per block.** Days 1–7 can
then use lags as short as 7, which is genuinely more information than lag-28 allows. Cost: 4× the
models and 4× the training time.

The pilot is the cheap place to test it, if it is needed at all. Decide on evidence, and record the
decision in `DECISIONS.md`.

## Gate

- [ ] All four candidates scored on all three folds with `src/metrics.py`, StatsForecast broken out per sub-model
- [ ] At least one clearly beats the `CA_1`-scope seasonal-naive WRMSSE (from `reports/baselines.csv`)
- [ ] Lumpy quadrant (18.0% of CA_1) checked for whether it drags the weighted score; if it does, a two-stage zero/non-zero or quantile-objective arm tried and scored
- [ ] `reports/pilot_scores.csv` written and committed
  (`model`, `fold`, `wrmsse`, `rmsse`, `wmae`, `bias_pct`, `runtime_s`)
- [ ] Per-model runtime measured and extrapolated to 10 stores
- [ ] Horizon strategy (single vs 4×7-day) decided and recorded
- [ ] Per-level and per-ADI/CV²-quadrant breakdown produced for the best model

## Invariants

- Folds end at `d_1913`; **the holdout is not touched in this phase** (invariant 1).
- Features come from `src/features.py`, not redefined in the notebook (see P5).
- Chronological splits only (invariant 5).

Full list: [README](README.md#invariants).
