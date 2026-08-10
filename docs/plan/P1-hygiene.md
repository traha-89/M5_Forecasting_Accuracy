# P1 — Load & hygiene check

**Output:** `notebooks/01_data_hygiene.ipynb` → `data/processed/sales_long.parquet`
**Effort:** 1 session
**Gate:** assertion suite passes; long-format parquet written with row count and dtypes recorded

## Objective

Load the raw CSVs with correct dtypes, prove the data is shaped the way we think it is, identify the
structural zeros that must not be modelled as demand, and persist a long-format panel every later
phase builds on.

## Assumes

- P0 passed: `pyarrow` available, `data/processed/` exists.

## Loading

Use explicit dtypes from the start. A naive `read_csv` on the wide sales file costs several GB in
object dtype for no benefit; typed, it is around 120 MB.

- Day columns (`d_1`…`d_1941`) → `int16`
- `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` → `category`
- `sell_price` → `float32`

Record memory usage before and after downcasting in the notebook.

## Structural assertions

Assert, don't eyeball — these should fail loudly if the data is not what we expect:

- 30,490 unique `id` values; `id` decomposes exactly into `item_id` + `_` + `store_id` + `_evaluation`
- `sales_train_evaluation.csv` and `sales_train_validation.csv` agree on `d_1`–`d_1913`
  (after which validation is redundant and we work only from the evaluation file)
- `calendar.csv` has 1,969 rows, contiguous dates from 2011-01-29 to 2016-06-19, no gaps
- `wm_yr_wk` maps consistently to date (one week value per 7 consecutive days)
- 10 stores × 3,049 items; 3 states, 3 categories, 7 departments
- `sample_submission.csv` has 60,980 rows in two id blocks of 30,490

## Content checks

- **Negative or implausible sales** — expect none. Flag loudly if present.
- **Price coverage.** A missing `sell_prices` row means the item was not on sale in that store that
  week. This could in principle be pre-release *or* a mid-life gap (stockout, delisting) — don't
  assume it's pre-release without checking. `sell_prices` is a weekly catalog/price table, not
  inventory, so a stockout would show `sales == 0` with the price row still present, not a missing
  row; but confirm coverage is one contiguous run per series (release week through the final
  calendar week, no interior gap, no early stop) before treating any gap as pre-release. Quantify
  the leading gap per series; this defines the release date and lets us drop ~12–13% of rows that
  are structural zeros, not demand signal.
- **Christmas closures.** Stores close 25 December; every series should read zero on those five dates
  (2011–2015). Confirm this, and mark the dates so they are not modelled as demand collapse.
- **Sparsity profile** — zero fraction overall and by category, store, department. Expect ~68% overall
  (an external planning-time prior, not derived from this repo's own data — see `DECISIONS.md`,
  pre-P0 section, "Note on the `~68%` zero-rate figure").
- **Dead series** — items with no sales in the final 60 days. Count them and record the number; the
  handling decision (forecast zero vs let the model decide) is made in P7, not here.
- **Event columns** — nulls are meaningful ("no event"), not missing data. Do not fill them.
- **Duplicate price rows** — expect none on (`store_id`, `item_id`, `wm_yr_wk`).

## Output

Melt to long `(id, d, sales)`, join `calendar.csv` on `d` and `sell_prices.csv` on
(`store_id`, `item_id`, `wm_yr_wk`), drop pre-release rows, write parquet.

Expected row counts — verify against these:

- Full melt: 30,490 × 1,941 = **59,181,090 rows**
- After the pre-release cut: roughly **46–47M rows**

Record the actual post-cut count in `DECISIONS.md`; later phases use it as a sanity check.

## Gate

- [x] All structural assertions pass
- [x] `data/processed/sales_long.parquet` written
- [x] Row count, dtypes, and file size recorded in `DECISIONS.md`
- [x] Christmas closure dates and dead-series count recorded

Passed 2026-08-09. See `docs/plan/DECISIONS.md` (`## P1 — Load & hygiene check`) for the recorded
evidence, including two deviations from this brief's stated expectations: the Christmas-closure
check uses a 0.1% tolerance instead of a hard zero, and the actual pre-release drop is ~20.8% (not
this brief's "~12-13%" estimate) — the drop matches the brief's own "~46-47M rows post-cut" target
exactly, so the "~12-13%" figure above is a documentation error, not a pipeline defect.

## Invariants

- Sales `int16`, prices `float32`, ids `category` (invariant 3).
- Do not compute anything over `d_1914`–`d_1941` beyond the structural assertions above — no summary
  statistics, no plots (invariant 1). The hygiene check covers the training range.

Full list: [README](README.md#invariants).
