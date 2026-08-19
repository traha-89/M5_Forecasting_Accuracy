# P4 — Baselines

**Output:** `notebooks/04_baselines.ipynb`
**Effort:** ~half a session
**Gate:** a scored baseline table — the number every model must beat

## Objective

Establish the bar. Without baselines, a WRMSSE of 0.7 means nothing; with them, it either beats
seasonal-naive or the modelling has not earned its complexity.

## Assumes

- P3 passed: `src/metrics.py` importable and validated.
- P1 passed: long-format parquet exists.

## Validation folds

Score on **rolling-origin folds**, the same ones P6 and P7 use, so comparisons stay apples-to-apples:

| Fold | Trains on | Predicts | `training_end` | Weight window |
|---|---|---|---|---|
| 1 | `d_1` – `d_1829` | `d_1830` – `d_1857` | 1829 | `d_1802` – `d_1829` |
| 2 | `d_1` – `d_1857` | `d_1858` – `d_1885` | 1857 | `d_1830` – `d_1857` |
| 3 | `d_1` – `d_1885` | `d_1886` – `d_1913` | 1885 | `d_1858` – `d_1885` |

Pass `training_end`, `weight_start` and `weight_end` explicitly on every call. `src/metrics.py`
defaults to the holdout's window (`d_1886`–`d_1913`); left at the default, folds 1 and 2 are scored
using days after their own training cutoff.

Never a random split — shuffling a time series leaks the future into the past and produces a
beautiful, meaningless score.

## Two scopes

Score every baseline twice:

- **`full`** — all 30,490 series, all 12 aggregation levels.
- **`CA_1`** — the 3,049 CA_1 series only, which is what P6's pilot can produce. On one store the
  12 levels collapse (Total = State = Store; levels 6–9 into 4–5; levels 10–12 into each other), so
  score the 4 distinct levels that survive and record which they are.

P6 compares against the `CA_1` rows; P7 against `full`. A pilot score compared against a full-panel
bar is not a comparison.

## The five baselines

| Baseline | Definition | What it tells us |
|---|---|---|
| Naive | Repeat the last observed day for all 28 | Floor. Anything losing to this is broken. |
| Seasonal naive | Same weekday, one week back | **The real bar** — captures the strongest signal in the data for free. |
| Moving average 28 | Mean of last 28 days | Level without seasonality; isolates how much day-of-week is worth. |
| Weekday mean | Mean of that weekday over last 8 weeks | Level + seasonality, still trivial. Often surprisingly competitive. |
| All zeros | Predict 0 everywhere | Sanity anchor. On sparse series this scores better than intuition suggests, and it exposes metric bugs fast. |

## Output

Write **`reports/baselines.csv`** with one row per baseline per fold per scope:
`baseline`, `fold`, `scope`, `wrmsse`, `rmsse`, `wmae`, `bias_pct`. This file is committed — P6 and P7
load it rather than re-running the baselines.

Also write **`reports/dead_series.csv`** — the `id` of every series with zero sales in
`d_1854`–`d_1913`. P7 sets the handling policy from it and P8's pre-flight checks against it.

Naive-baseline errors and weights depend only on the fold, not on the baseline being scored. Compute
them once per fold and reuse across all five; recomputing per baseline turns a minutes-long phase
into an hours-long one.

Also summarise in the notebook and copy to `DECISIONS.md`:

| Baseline | WRMSSE (mean over folds) | RMSSE | WMAE | Bias % |
|---|---|---|---|---|

Record the **seasonal-naive WRMSSE explicitly** — it is the number quoted in every later phase as the
bar to beat.

## Gate

- [ ] All five baselines scored on all three folds, at both `full` and `CA_1` scope
- [ ] Fold-local `training_end` / weight windows used, not the module defaults
- [ ] `reports/baselines.csv` and `reports/dead_series.csv` written and committed
- [ ] Seasonal-naive WRMSSE recorded in `DECISIONS.md`, both scopes
- [ ] All-zeros produces a finite score (re-confirms the P3 metric implementation)
- [ ] Per-level WRMSSE reported for at least seasonal-naive, as a reference shape

## Invariants

- Folds end at `d_1913` or earlier; the holdout is not scored in this phase (invariant 1).
- Chronological splits only (invariant 5).

Full list: [README](README.md#invariants).
