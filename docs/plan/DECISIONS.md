# Decisions log

Append-only. At each phase gate, record **what was decided and the number that justified it.**

This is what carries context between sessions. Without it, later phases re-derive choices
inconsistently — which is how the pilot and the full run end up with different feature definitions
and an invalid comparison. Keep entries short; link to the notebook or figure for detail.

Format:

```
## P<n> — <phase name>            (YYYY-MM-DD)
- **Decision:** …
- **Evidence:** … (number, figure, or notebook cell)
```

---

## Pre-P0 — plan decisions (2026-08-08)

Made when the plan was drafted, before any code.

- **Model shortlist:** LightGBM Tweedie, LightGBM Poisson, Nixtla StatsForecast ensemble, XGBoost.
  **Rationale:** best accuracy-per-hour on a CPU-only 8-core box; the Tweedie/Poisson pair isolates
  the objective's contribution, XGBoost guards against a LightGBM-specific artifact, and
  StatsForecast provides the only per-series arm for contrast.
- **Sequencing:** pilot the full pipeline on store `CA_1`, then scale to all ten.
  **Rationale:** iteration on 3,049 series takes minutes; on 30,490 it takes hours. Design errors get
  caught cheaply.
- **Forecast granularity:** global gradient-boosted models **partitioned by store** (10 models,
  ~3,049 series each), not one model per series.
  **Rationale:** per-series fitting discards transferable structure (SNAP, events, price response) and
  30,490 fits is infeasible here; the store seam is principled because SNAP schedules are
  state-specific and assortment differs by store. Also fits memory: ~1.4 GB per partition vs ~14 GB
  for all ten.
- **Deliverable form:** numbered Jupyter notebooks per phase, plus thin `src/` modules for metrics
  (P3) and features (P5) only.
  **Rationale:** those two are imported by four to five downstream notebooks each; copy-pasting them
  causes silent drift between the pilot and the full run.
- **Excluded metrics:** sMAPE and MAPE. **Rationale:** both divide by actuals; ~68% of this dataset is
  zero, so they are undefined or explosive.
- **Note on the `~68%` zero-rate figure** (used here and in `P1-hygiene.md`'s sparsity check,
  `P3-metrics.md`, `P6-pilot.md`, `plan.html`): stated as a planning-time expectation, not derived
  from an in-repo EDA — there is no notebook or commit predating the plan (`f4ca976`) that computed
  it from `data/`. It reflects the zero-rate commonly cited in published M5 competition analysis for
  this dataset's item×store×day granularity, carried in as a prior to validate against. P1
  independently measured 68.0% from the actual data (`DECISIONS.md`, P1 entry) and confirmed it, so
  the prior held — but treat the *planning-time* figure as an unverified external citation, not a
  locally-derived one, if it's ever relied on before P1 has run.

### Still open

- **Outlier policy** — decided in P2 on evidence. Note the tension: clipping promotional spikes will
  improve RMSSE while making the forecast less useful operationally.
- **Prediction intervals** — out of scope. M5 had a companion Uncertainty track using pinball loss
  across nine quantiles; LightGBM can produce these with quantile objectives at ~9× training cost.
- **Horizon strategy** — single non-recursive model with lag ≥28 is the plan; the 4×7-day block
  alternative is tested in P6 and decided there.
- **Segment routing** — whether to route different models to different ADI/CV² quadrants. Decided in
  P7, only if the breakdown supports it.

---

## P0 — Environment & scaffolding            (2026-08-09)

- **Decision:** `src/` import convention is an editable install (`pip install -e .`) backed by a
  minimal `pyproject.toml` declaring `src` as a package, not a per-notebook `sys.path` shim.
  **Evidence:** `import src` resolves correctly to `src/__init__.py` when run with `notebooks/` as
  the working directory, confirmed via `cd notebooks && python -c "import src; print(src.__file__)"`.
- **Decision:** directory skeleton (`notebooks/`, `src/`, `tests/`, `reports/figures/`, and gitignored
  `models/`, `data/interim/`, `data/processed/`) created as specified in P0-setup.md.
  **Evidence:** all four model libraries import cleanly, `pytest --version` succeeds, `M5` kernel
  registered in `jupyter kernelspec list`, `from src import …` verified from `notebooks/`.
- **Installed versions:** lightgbm 4.7.0, xgboost 3.2.0, statsforecast 2.1.1, pyarrow 25.0.0,
  scikit-learn 1.9.0, pandas 2.3.3, numpy 2.4.6. No source builds required — all installed from
  wheels on Windows/Python 3.11.
- **Note:** installing `statsforecast` downgraded `pandas` from 3.0.5 to 2.3.3 to satisfy its pin.
  No other packages in this repo depend on pandas 3.x features, so left as resolved by pip.

## P1 — Load & hygiene check            (2026-08-09)

- **Decision:** relax the Christmas-closure check from a hard "zero sales on 25 Dec, every series"
  assertion to a bounded tolerance (fail if non-zero cells exceed 0.1% of the 5 × 30,490 checked).
  **Evidence:** 2011–2015 each show 8–17 non-zero rows (all small counts, 1–6 units, all
  `FOODS_3_*` items, scattered across CA/TX/WI stores) — 60 non-zero cells out of 152,450 checked,
  0.039% overall. A known minor M5 data quirk, not a loading/dtype bug: confirmed by inspecting the
  offending rows directly (`notebooks/01_data_hygiene.ipynb`, Christmas closures cell). The 5 dates
  are marked as closures regardless so they aren't modelled as demand collapse; the tolerance only
  changes whether the *assertion* hard-fails, not how the dates are treated downstream.
- **Note:** P1-hygiene.md's own two stated expectations for the pre-release cut are mutually
  inconsistent — "~12-13%" dropped implies a post-cut count of ~51.5M (59,181,090 × 0.87), but the
  same brief also states "roughly 46-47M rows" post-cut, which implies ~21-22% dropped. The actual
  run drops 12,299,413 of 59,181,090 rows (20.78%), landing at 46,881,677 rows — matching the row-
  count target exactly, not the percentage prose. Verified the drop is entirely explained by the
  per-series leading (pre-release) gap, with zero trailing/post-delisting gap, so this is a
  documentation inconsistency in the brief, not a bug in the melt/join/drop logic.
- **Decision:** `data/processed/sales_long.parquet` written — 46,881,677 rows, 227.7 MB, 22 columns.
  Schema matches the `docs/plan/README.md` data contract exactly (`id`/`item_id`/`dept_id`/`cat_id`/
  `store_id`/`state_id`/`weekday`/`event_name_1`/`event_type_1`/`event_name_2`/`event_type_2`
  `category`; `d`/`sales`/`wm_yr_wk`/`year` `int16`; `wday`/`month`/`snap_CA`/`snap_TX`/`snap_WI`
  `int8`; `date` `datetime64[ns]`; `sell_price` `float32`, no nulls post-cut).
  **Evidence:** full melt asserted at 59,181,090 rows exactly (30,490 × 1,941); post-join row count
  unchanged (both joins 1:1); post-cut count asserted in [45M, 48M]; per-column dtype assertions all
  pass. `notebooks/01_data_hygiene.ipynb`, Output section.
- **Dead series:** 955 of 30,490 (3.1%) have zero sales in the final 60 training days (`d_1854`-
  `d_1913`, deliberately not the file's literal last 60 columns, which would reach into the
  `d_1914`-`d_1941` holdout — see invariant 1). Handling decision deferred to P7 per the brief.
- **Sparsity:** 68.0% overall zero fraction, matching the brief's ~68% expectation.
- **Known issues (not yet fixed):** a code review after the gate passed found that the reported
  "no negative sales" max/percentile check and the sparsity-profile check (68.0% figure above) both
  compute over `DAY_COLS = d_1`-`d_1941`, which includes the `d_1914`-`d_1941` holdout — an invariant
  1 violation (contrast with the dead-series check two cells later, which correctly scopes to
  `d_1854`-`d_1913`). Separately, the Output section's `d` column depends on a `d_num` column mutated
  onto `calendar` inside the earlier price-coverage cell rather than computed locally, an implicit
  cross-cell dependency. None of these affect `sales_long.parquet`'s written schema or row count, and
  the gate above stands, but the two summary numbers should be treated as informational-only (not
  clean invariant-1 evidence) until fixed. Tracked in issues #9, #10, #11 — deferred to a later
  session, not blocking P2.
- **Strengthened post-gate (2026-08-10):** the price-coverage cell asserted "missing `sell_prices`
  row = pre-release" without checking for mid-life gaps (stockout, delisting), which would look
  identical to pre-release under the original leading-gap-only logic and get silently mis-dropped
  as a structural zero. Added an explicit check: for all 30,490 `(item_id, store_id)` combos with
  price data, coverage is one contiguous `wm_yr_wk` run from release week through the final
  calendar week — 0 interior gaps, 0 combos stopping early. Confirms the leading-gap cut is safe
  for this dataset; not a general guarantee for other data. Also added: `wm_yr_wk` truncated-week
  check now asserts any `<7`-day group falls only at the calendar's first/last week (previously
  `between(1, 7)` alone would have passed a truncated week anywhere); and an explicit full-
  cross-product check (every item present in every store exactly once), which was already implied
  by existing id-uniqueness + cardinality assertions but is now checked directly rather than left
  as an unstated consequence. `notebooks/01_data_hygiene.ipynb`, re-run clean end-to-end after each
  change.
- **Issue #9 fixed (2026-08-11):** the negative-sales check and the sparsity-profile check now
  compute over a new `TRAIN_DAY_COLS = d_1`-`d_1913` list instead of the full-range `DAY_COLS`,
  closing the invariant 1 violation recorded above. Re-running the notebook end-to-end shifted the
  sparsity figure from 68.0% (full range) to **68.2%** (training range only) — the by-category/
  store/department breakdowns shifted correspondingly by a few tenths of a point each; the
  negative-sales max (763) and 99.9th percentile (47.0) were unchanged, since the holdout range
  didn't contain a more extreme value. `sales_long.parquet`'s schema and row count are unaffected
  (those are built from the full melt/join/drop pipeline, not from these two checks). Issue #10
  (the `d_num` cross-cell dependency) was not removed — `d_num` is still mutated onto `calendar` in
  the price-coverage cell and consumed later in the Output section's join — but it is now called
  out explicitly in that join cell's comment rather than being an unstated dependency. Issue #11
  not addressed in this pass. `notebooks/01_data_hygiene.ipynb`, cells `02121c63`, `96baedf7`,
  `d84bd04b`, `7b77e394`, `97feb64d`.
- **Issue #11 fixed (2026-08-11):** the Output section's `d` column no longer depends on the
  `calendar["d_num"]` mutation made in the earlier price-coverage cell. Two changes: (1) the
  price-coverage cell (`d08afdfd`) now computes a local `calendar_d_num` Series instead of
  assigning `calendar["d_num"] = ...` — `calendar` itself is never mutated, so no later cell can
  come to depend on this one having already run; (2) the Output join cell (`fc3a2b39`) computes
  its own `d` column directly from `sales_long["d_label"]` (`str.replace("d_", "") .astype("int16")`)
  instead of renaming a `d_num` column carried over from `calendar`. This closes issue #10's
  cross-cell dependency as well as #11 — the two were describing the same underlying pattern from
  different angles. Re-ran the notebook end-to-end: all outputs identical to the pre-fix run
  (59,181,090-row melt, 12,299,413 pre-release rows dropped, 46,881,677-row / 227.7 MB parquet,
  schema check OK, `d` still `int16`) — this was a pure refactor of *how* `d` gets computed, not a
  change to its values. `notebooks/01_data_hygiene.ipynb`, cells `d08afdfd`, `fc3a2b39`.

## P2 — Exploratory analysis (in progress)

- **Deviation (2026-08-12):** `docs/reference/M5-Competitors-Guide.pdf` committed in PR #20
  alongside the in-progress EDA work, rather than held back until the P2 gate passes as originally
  decided. **Rationale:** user explicitly requested it be added to this PR now; no functional
  reason to keep withholding it once the branch/PR already exist and cite it. Gate itself is
  unaffected — still requires the four checklist items in `P2-eda.md` before merge.

<!-- Append phase entries below as gates are passed. -->
