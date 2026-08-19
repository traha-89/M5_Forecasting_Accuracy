# P5 — Feature engineering

**Output:** `notebooks/05_features.ipynb` + `src/features.py`, per-store parquet partitions
**Effort:** 1–2 sessions
**Gate:** CA_1 features build in <5 min under 8 GB peak RAM; every feature justified against leakage

## Objective

Build the feature matrix the gradient-boosted models train on, as ten per-store parquet partitions.

This phase writes `src/features.py` as well as the notebook. P6 and P7 both import it — if the pilot
and the full run use different feature definitions, every comparison between them is invalid, and
that bug is very hard to see.

## Assumes

- P1 passed: long-format parquet with calendar and prices joined.
- P2 passed: the feature-hypothesis table exists and should drive what is built here.

## The lag-28 constraint

We forecast 28 days ahead in one shot, so on the last day of the horizon the most recent *observed*
sale is 28 days old. **Every lag and rolling window is therefore anchored at lag 28 or longer.**

This gives a single non-recursive model valid across the whole horizon — no error compounding, no
recursive prediction loop — and it is the design that proved most robust in the actual competition.
A lag-7 feature will look excellent in training and be unavailable at prediction time.

## Feature families

| Family | Features |
|---|---|
| Calendar | `wday` (7-level categorical), `month`, `year`, day-of-month, `sin`/`cos` of day-of-year, `is_closure_day` |
| Events | `event_name_1/2` as categoricals; days-to and days-since nearest event |
| SNAP | snap flag **matched to the series' own state**, not all three columns |
| Price | `sell_price`; `relative_price` (price ÷ item's rolling mean); price ÷ item's historical max; price-change flag; weeks since last price change; `never_repriced` |
| Lags | sales at lag 28, 29, 30, 35, 42, 49, 56, 364 |
| Rolling | mean / max over 7, 14, 30, 60, 180 windows and std over 30, 60, 180, **all computed on lag-28 sales**; non-zero-conditional mean over 30, 60, 180; same-weekday rolling mean |
| Intermittency | days since last non-zero sale; non-zero count in last 28 / 60; mean sales conditional on non-zero |
| Lifecycle | `days_since_release`, `history_days` — per series, not panel-relative |
| Identity | `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` as native LightGBM categoricals |
| Encodings | mean and non-zero-conditional mean by item, item×store, dept×store, cat×wday — **fit on training folds only**, so exposed as `build_encodings(train_slice)` and *not* written into the store partitions |

Roughly 60 features per store. Deviations from this list go in `DECISIONS.md`.

## Build rules

- **NaN stays NaN.** 1.36% of series have under 365 days of history, so `lag_364` and the 180-day
  rolling windows are undefined for them. LightGBM and XGBoost handle NaN natively; a zero-fill
  asserts "no sales a year ago", which is false.
- **Do not clip or winsorize the target** (P2 outlier policy). Rolling `max` is spike-driven by
  design.
- **Do not join `quadrant`/`adi`/`cv2` as features.** They are static full-history statistics and
  redundant with the Intermittency family; `series_segments.parquet` is a P6/P7 evaluation grouping
  key only.
- **Keep both `day-of-month` and the SNAP flag** despite their collinearity — there is a residual
  non-SNAP start-of-month effect.
- **No global time index.** A monotone counter is saturated at its terminal split for every
  forecast day, so it cannot inform the horizon; use the per-series lifecycle features instead.

## Memory discipline

Non-negotiable on 15.9 GB:

- Sales `int16`, all engineered features `float32`, ids `category`
- **Build features per store, writing one parquet partition each** —
  `data/processed/features/store_id=CA_1/…`
- One store partition at ~60 float32 features is roughly **1.4 GB** — comfortable.
  All ten at once is roughly **14 GB** — not.
- Delete intermediate frames and `gc.collect()` between stores.

Measure and record peak RAM for the CA_1 build. If it exceeds 8 GB, the full run will not complete
and the feature set needs trimming before P6.

## Leakage review

Every feature must be answerable: *would this value be knowable at prediction time?*

- Lags/rolling: yes if anchored ≥28 (see above).
- Calendar/events/SNAP: yes — `calendar.csv` covers the forecast window through `d_1969`.
- Price: yes for the horizon, using the last observed weekly price. Confirm the forward-fill rule.
- Encodings: **only if fit on the training fold**, not the full series. This is the most common
  leakage source in the whole project, and why they are computed at fit time rather than stored.

Document this check per family in the notebook — it is a gate item.

## Gate

- [ ] CA_1 partition builds in <5 min, peak RAM <8 GB (both measured and recorded)
- [ ] Leakage review documented per feature family
- [ ] All lags and rolling windows confirmed ≥28
- [ ] `src/features.py` importable, exposing `build_features` and `build_encodings`; notebook imports rather than redefines
- [ ] Partitions confirmed to contain no encoding columns
- [ ] Feature count and partition sizes recorded in `DECISIONS.md`

## Invariants

- Lag ≥ 28 on everything (invariant 2) — the defining constraint of this phase.
- Per-store partitions, `float32`/`int16`/`category` (invariant 3).
- Encodings fit on training folds only (invariant 4).
- No feature computed over `d_1914`–`d_1941` (invariant 1).

Full list: [README](README.md#invariants).
