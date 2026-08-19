# P8 — Final forecast

**Output:** `notebooks/08_final_forecast.ipynb`, `submission.csv`, README write-up
**Effort:** 1 session
**Gate:** 60,980 rows, non-negative, no NaN, totals plausible against recent actuals

## Objective

Produce the deliverable: a 28-day forecast for `d_1942`–`d_1969` (2016-05-23 → 2016-06-19) for all
30,490 item–store series.

## Assumes

- P7 passed: model selected, holdout WRMSSE reported, model card written.

## Refit

Refit the selected model on the **full `d_1`–`d_1941` history**, now including the holdout period.
This is legitimate — model selection is finished, so there is nothing left to leak into. Using the
most recent 28 days of actuals meaningfully helps a 28-day-ahead forecast.

Build features for the forecast window from:

- `calendar.csv`, which covers through `d_1969` — this is why it extends past the sales data
- The last observed weekly price per (`store_id`, `item_id`), forward-filled across the horizon

All lags are ≥28, so every feature for `d_1942`–`d_1969` is computable from observed data with no
recursion. Confirm this rather than assuming it.

The window's events — MemorialDay, NBA Finals, Ramadan start, Father's Day — appear in no CV fold and
no holdout, so the event features are unvalidated here. Note it in the write-up. The closure flag is
inert in this window.

## Submission format

`sample_submission.csv` has **60,980 rows in two blocks**:

| Block | id suffix | Rows | Wants |
|---|---|---|---|
| 1 | `_validation` | 30,490 | `d_1914` – `d_1941` |
| 2 | `_evaluation` | 30,490 | `d_1942` – `d_1969` |

We can fill both — the `_validation` block from the P7 holdout predictions, the `_evaluation` block
from this phase's forecast. Columns are `F1`–`F28`. Preserve the template's exact id order.

## Pre-flight checks

Run all of these before writing the file:

- [ ] Exactly 60,980 rows, in the template's id order
- [ ] All predictions finite and ≥ 0
- [ ] No NaN in any `F1`–`F28` column
- [ ] Forecast totals within a plausible band of the last 28 observed days, **per category** — the
      comparison window (Apr 25–May 22) and the forecast window (May 23–Jun 19) sit at different
      points on an annual curve that swings 18.5 pct pts for HOUSEHOLD, and catalog growth pushes
      totals up while per-series velocity falls, so flat-to-mildly-up is the expected shape
- [ ] FOODS forecasts lift on the June 1–15 SNAP days, per state — SNAP explains 86.7% of all
      spike-days with a known driver, so a flat SNAP response means a mis-joined state flag
- [ ] No series forecast flat-zero unless it is genuinely dead (cross-check `reports/dead_series.csv`)
- [ ] The `F1`–`F28` profile shows the expected weekly shape, not a suspiciously smooth line —
      plot a sample of series and look at them

That last one catches more real bugs than the rest combined.

## Write-up

Update `README.md` with: approach, final holdout WRMSSE against the baselines, what worked, what
didn't, and what would be tried next. Link to `docs/plan/` and the model card.

## Gate

- [ ] All pre-flight checks pass
- [ ] `submission.csv` written with both blocks filled
- [ ] Sample of forecast series plotted and visually inspected
- [ ] README write-up complete

## Invariants

- Invariant 1 is now satisfied and retired: selection is complete, so refitting on `d_1..d_1941` is
  correct. Do not, however, revisit model choice after seeing this forecast.
- Lag ≥ 28 still applies to the forecast-window features (invariant 2).
- Per-store partitions for the refit (invariant 3).

Full list: [README](README.md#invariants).
