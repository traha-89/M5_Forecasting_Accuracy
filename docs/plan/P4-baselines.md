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

| Fold | Trains on | Predicts |
|---|---|---|
| 1 | `d_1` – `d_1829` | `d_1830` – `d_1857` |
| 2 | `d_1` – `d_1857` | `d_1858` – `d_1885` |
| 3 | `d_1` – `d_1885` | `d_1886` – `d_1913` |

Never a random split — shuffling a time series leaks the future into the past and produces a
beautiful, meaningless score.

## The five baselines

| Baseline | Definition | What it tells us |
|---|---|---|
| Naive | Repeat the last observed day for all 28 | Floor. Anything losing to this is broken. |
| Seasonal naive | Same weekday, one week back | **The real bar** — captures the strongest signal in the data for free. |
| Moving average 28 | Mean of last 28 days | Level without seasonality; isolates how much day-of-week is worth. |
| Weekday mean | Mean of that weekday over last 8 weeks | Level + seasonality, still trivial. Often surprisingly competitive. |
| All zeros | Predict 0 everywhere | Sanity anchor. On sparse series this scores better than intuition suggests, and it exposes metric bugs fast. |

## Output

A table, copied to `DECISIONS.md`:

| Baseline | WRMSSE (mean over folds) | RMSSE | WMAE | Bias % |
|---|---|---|---|---|

Record the **seasonal-naive WRMSSE explicitly** — it is the number quoted in every later phase as the
bar to beat.

## Gate

- [ ] All five baselines scored on all three folds
- [ ] Seasonal-naive WRMSSE recorded in `DECISIONS.md`
- [ ] All-zeros produces a finite score (re-confirms the P3 metric implementation)
- [ ] Per-level WRMSSE reported for at least seasonal-naive, as a reference shape

## Invariants

- Folds end at `d_1913` or earlier; the holdout is not scored in this phase (invariant 1).
- Chronological splits only (invariant 5).

Full list: [README](README.md#invariants).
