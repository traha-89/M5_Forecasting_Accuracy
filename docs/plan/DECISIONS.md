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
  **Resolved 2026-08-13** — see the "P2 — Exploratory analysis" section below: no clipping.
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
- **Validation added (2026-08-14):** pre-release rows (missing `sell_price`) all have `sales == 0`.
  Confirms that the leading-gap cut removes only structural zeros, not actual demand mismatches.
  **Evidence:** all 12,299,413 pre-release rows verified to have sales = 0; assertion added to
  the Output section. `notebooks/01_data_hygiene.ipynb`, cell `bf1e3668`.

## P2 — Exploratory analysis            (2026-08-13)

- **Deviation (2026-08-12):** `docs/reference/M5-Competitors-Guide.pdf` committed in PR #20
  alongside the in-progress EDA work, rather than held back until the P2 gate passes as originally
  decided. **Rationale:** user explicitly requested it be added to this PR now; no functional
  reason to keep withholding it once the branch/PR already exist and cite it. Gate itself is
  unaffected — still requires the four checklist items in `P2-eda.md` before merge.
- **Decision (2026-08-13) — outlier handling policy:** do not clip or winsorize the training
  target for extreme single-day spikes; carry them as-is into P5/P6.
- **Evidence:** per-series IQR fence on non-zero sales days (`upper_fence = Q3 + 1.5*(Q3-Q1)`)
  flags 1,008,763 spike-days (2.19% of rows; 99.4% of series have >=1). Magnitudes are almost all
  mild — median 1.45x the fence, 75th pct 2x. Only 274 days (0.0006% of rows, 193/30,490 series,
  0.63%) exceed 10x the fence, and the 10 most extreme are plausible large integer counts (e.g.
  601 units on an item whose typical non-zero day is ~3-4), not fractional/negative values — P1's
  hygiene checks already screened for those separately, so nothing here reads as a data error.
  43.2% of spikes align with a known driver already built in this notebook (SNAP 378,445; event
  76,817; price-drop 18,500 — SNAP dominates, consistent with the Price section's finding that
  price cuts are rare); the remaining 56.8% are unexplained by those three signals but, given how
  mild most magnitudes are, read as ordinary right-skewed demand variability rather than errors.
  `notebooks/02_eda.ipynb`, Outliers section, figs `outliers_spike_drivers.png`,
  `outliers_magnitude_distribution.png`.
- **Rationale:** clipping would lower in-sample RMSSE (the brief's own explicit warning) while
  discarding real demand signal the model needs to learn to predict occasional large days; the
  extreme tail is both negligible in volume and plausible in magnitude, so there's no evidence of
  measurement error to justify correcting values as opposed to genuine but rare demand. The
  SNAP/event/price-drop features already built in Price/Events/SNAP give the model a route to
  anticipate the explainable 43% of spikes; no additional feature is proposed for the unexplained
  majority — if a specific series' RMSSE later turns out dominated by an unexplained spike, that's
  a case for a per-series review at the modeling phase, not a blanket EDA-stage correction.
- **Hypothesis table (2026-08-13)**, copied from `notebooks/02_eda.ipynb`'s Hypothesis table
  section (14 rows, gate requires >=10):

| # | Finding | Evidence | Proposed feature | Verdict |
|---|---|---|---|---|
| 1 | Raw trend (+81.3%) is mostly assortment growth (active series +152.9%); units per active series actually fell -36.7% | fig `trend_assortment.png` | Reject a raw-total trend/momentum feature; adopt one normalized against active-catalog size instead | reject (raw) / adopt (normalized) |
| 2 | Weekend lift is strong in every category and fairly uniform across categories (HOUSEHOLD +33.5, FOODS +27.7, HOBBIES +24.0 pct pts of trend); Friday behaves like a transition day, not an ordinary weekday | fig `seasonality_dow_mstl.png` | Full 7-level `wday` categorical, not a binary weekend flag | adopt |
| 3 | Annual seasonal swing differs sharply by category: HOUSEHOLD (18.5 pct pts, Aug peak/Dec trough) is ~1.7x FOODS/HOBBIES (10.9/11.1) | fig `seasonality_month_mstl.png` | Per-category month-of-year / annual-harmonic feature, not pooled | adopt |
| 4 | FOODS SNAP lift is state-specific and varies ~10x: WI +2.5, TX +2.0, CA +0.2 pct pts of trend | fig `snap_lift_by_state.png` | Per-state SNAP flag matched to series `state_id` (not a single pooled `is_snap_day`) | adopt |
| 5 | Closure holidays (Christmas -11.9%, Thanksgiving -8.4%, Easter -8.4%) dip on the day with a lead-up lift beforehand (Thanksgiving +15.1%); open-store federal holidays (Labor Day +12.5%) lift instead | fig `events_headline_phases.png` | Per-`event_name` dummy plus explicit lead-up/hangover offset dummies for the closure-holiday group | adopt |
| 6 | Pooling by `event_type` still can't replace per-event dummies: mean-absolute effect (fixing the earlier signed-mean's dip/lift cancellation) shows `National` has the largest typical magnitude (5.17%), but the aggregate can't say which direction any given National event moves sales | fig `events_by_type.png` | A single pooled `event_type` feature | reject |
| 7 | Price changes are rare: median 0.7% of weeks change, 27.5% of series never change price, median gap ~91 weeks between changes | fig `price_change_frequency.png` | A price-change-recency / "still at original price" structural-break flag | adopt |
| 8 | Sales response to a price change is noise-dominated at the pooled level (Spearman corr with normalized sales: raw price -0.028, relative-to-baseline price +0.024 - both near zero, one sign-flipped) | fig `price_change_sales_response.png` | A pooled cross-sectional price-elasticity coefficient | reject |
| 9 | 91% of series fall in the sparse ADI/CV² quadrants (intermittent 72.7%, lumpy 18.4%); only 6.1% are smooth | fig `intermittency_adi_cv2_quadrants.png` | Segment label from `series_segments.parquet` used for model routing (esp. isolating the 18.4% lumpy group) | adopt |
| 10 | Median series is zero-sales on 63.5% of days - mean/std-based statistics are unstable at this zero-inflation level | fig `intermittency_zero_fraction.png` | Any per-series scaling/normalization feature should use robust statistics (median/MAD or non-zero-conditional), not mean/std; favors a zero-inflated/Tweedie-style model objective | adopt |
| 11 | 64.1% of series joined the panel after `d_1` (only 35.9% present at panel start), matching the assortment-growth finding at the item level | fig `lifecycle_release_dates.png` | `days_since_release` / `history_days` feature | adopt |
| 12 | Only 0.6% of series (173) show a long dormant tail (>=180d trailing zero sales); no series ever drops out of the panel | fig `lifecycle_trailing_zero_run.png` | `days_since_last_sale` feature; a dedicated "discontinued series" model path | adopt (feature) / reject (dedicated path - group too small) |
| 13 | 0 series have <28 days of history (lag-28 always computable); 1.36% have <365 days, affecting annual-seasonal feature reliability | fig `lifecycle_history_length.png` | `history_days` used to gate/down-weight annual-seasonality features for short-history series | adopt |
| 14 | IQR-flagged spikes are common but almost all mild (median 1.45x fence); the extreme tail (>10x, 274 days / 0.0006% of rows) is plausible in magnitude, not a data-error signature; 43.2% of spikes align with an existing SNAP/event/price-drop flag | figs `outliers_spike_drivers.png`, `outliers_magnitude_distribution.png` | Clip/winsorize the training target for spikes | reject (see outlier policy above) |

- **Readability refactor (2026-08-14):** every code section (Trend, Seasonality, SNAP, Events,
  Price, Intermittency, Lifecycle, Outliers) split so each cell has one job — prepare data, or
  plot/report it — per the new "Notebook conventions" section in `CLAUDE.md`. Verified section by
  section via a line-diff against the pre-refactor commit (`324d85f`): every meaningful code line
  (asserts included) accounted for, with one exception traced to a deliberate removal (the
  now-redundant `sales_long["d"].max() <= TRAIN_END` assertion, since the parquet `filters`
  argument already enforces the training-range cut at read time). Caught and fixed one real bug
  in the process: the Trend/assortment data-prep cell was missing `pct_change_units`/`_series`/
  `_per_series`, which the plot cell referenced — would have raised `NameError` on a fresh run.
- **Issues #22, #23, #24 fixed (2026-08-14)** — all three were latent robustness/perf gaps found
  by a prior `/code-review` pass, none triggered by this dataset:
  - **#22:** the Intermittency section's nonzero-count assert (`a15b5bd2`) compared `n_nonzero`
    (0 for an all-zero series) against `n_nonzero_check` (`NaN` for the same series, since it's
    built from an `is_nonzero`-filtered groupby) — `0 == NaN` is `False`, so the assert would
    crash if an all-zero series ever appeared. Fixed with `.fillna(0)` before the comparison.
  - **#23:** the SNAP section's per-state lift calculation (`fa90545e`) used `grp.loc[0]`/
    `grp.loc[1]` after a groupby that only produces labels present in the data — would raise
    `KeyError` if a state's SNAP flag were ever constant over the aggregated window. Fixed with
    `.reindex([0, 1])` after the groupby.
  - **#24:** the Outliers section's `is_snap` lookup (`810066cd`) used a Python-level row-wise
    `.apply(..., axis=1)` over ~1M spike-day rows. Vectorized via a `(snap_CA, snap_TX, snap_WI)`
    matrix indexed by each row's state column position (`to_numpy()[np.arange(n), col_idx]`).
  - **Evidence:** full headless re-execution (`jupyter nbconvert --execute`) completed with exit
    code 0, zero cell errors. Cross-checked printed figures against this file's existing
    hypothesis-table/outlier-policy numbers post-fix — all identical (SNAP lift WI +2.52/TX
    +1.95/CA +0.23 pct pts; ADI/CV² quadrants 72.7%/18.4%/6.1%/2.8%; 1,008,763 spike days, 43.2%
    explained, driver counts 378,445 SNAP / 76,817 event / 18,500 price-drop) — confirming the
    fixes are pure edge-case guards / a perf change, not a behavior change on this dataset.
    `notebooks/02_eda.ipynb`, cells `a15b5bd2`, `fa90545e`, `810066cd`.

### SNAP effect size — units fix and scope decision (2026-08-14)

- **Units bug (fixed).** `snap_lift_df` held raw `resid / trend` ratios (WI 0.0252) while the column
  was named `snap_lift_pct_pts` and the plot's y-axis read "pct points of trend" — off by 100×. The
  prose had always quoted the correctly-converted values (+2.5 pct pts), so no published number was
  wrong, but the figure was. Converted at the point of computation; cell 27's figure regenerated.
  Same class of bug as the Seasonality section's, so worth watching for on any `resid / trend` ratio.
- **Day-of-month confound (investigated, not carried).** SNAP days are a fixed function of
  day-of-month (all within days 1–15), and `MSTL(periods=(7, 365))` has no monthly component, so a
  generic start-of-month effect sits in the residual alongside the SNAP effect. Two checks were run
  and then **removed from the notebook**: (a) a placebo estimate on HOBBIES/HOUSEHOLD, which SNAP
  cannot fund — lift was 7–10× smaller than FOODS (CA 0.04/0.10, TX 0.29/0.28, WI −0.38/−0.18),
  so contamination is small; (b) a cross-state difference-in-differences using the states'
  differing calendars — CA −0.21, TX +3.63, WI +5.19, same ranking as the naive estimate but
  roughly double the magnitude for TX/WI, because the two estimators average over different
  disbursement days.
- **Why removed:** neither check changes the P5 feature. The decision that mattered — a per-state
  `snap_<state>` flag rather than a pooled `is_snap_day` — follows from the schedules differing by
  state (`snap_schedule.png`), not from any effect size, and LightGBM estimates the per-state
  magnitude itself. Effect-size precision would matter for a parametric model with a hand-coded
  coefficient; it does not here. Hypothesis-table row 4 was softened accordingly: the earlier
  "varies ~10x" claim overstated what the naive estimate can support, and CA (+0.2) is not
  distinguishable from zero. No significance testing was done, by the same reasoning.
- **Do not redo this** unless a later phase needs a calibrated SNAP effect size rather than a flag.

### Trend section — removed raw by-level rolling average (2026-08-18)

- **Removed (post-gate).** `02_eda.ipynb`'s Trend and level section plotted a raw 28-day rolling
  average of units split by state/store/category/department (`by_level_data`, saved as
  `trend_by_level.png`). On review it added no decision-relevant information: it carries the same
  demand-vs-assortment ambiguity as the pooled MSTL trend earlier in the same section, its own
  interpretation text never cited it, and the by-level claim in hypothesis-table row 1 ("universal
  across every state/store/category/department") is already fully supported by the
  units-per-active-series growth check (`trend_by_level_growth.png`) that stays in the notebook.
  Scale differences between groups don't need a separate feature either — categorical identity
  (`state_id`/`store_id`/`cat_id`/`dept_id`, already planned for P5) lets the tree learn each
  group's baseline level directly from the split. Removed per the CLAUDE.md EDA-scope rule: does
  this change what gets built? A markdown note was left in the notebook at the removal point
  recording that this was reviewed and dropped.
- **P2 gate unaffected.** The gate passed 2026-08-13 citing `trend_assortment.png` for hypothesis
  row 1, not `trend_by_level.png` — nothing in the passed gate depended on the removed figure, and
  the brief's "split by state/store/category/department" requirement (`P2-eda.md` line 28) is still
  satisfied by the growth-by-level cells that remain. No change needed to `P2-eda.md`.

### Events section — headline event selection (2026-08-18)

- **What changed.** `P2-eda.md`'s Events bullet names Super Bowl, Thanksgiving, Easter, and
  Christmas as the events expected to "behave very differently from one another." That was a
  pre-EDA hypothesis about which events would be worth contrasting, not a final selection — once
  the full 30-event `event_effect_df` table existed, all four named events turned out to share the
  *same* behavioral pattern (event-day dip, +lead-up, closure/gathering holidays). No event
  showing the opposite pattern (event-day lift, stores open — LaborDay/MemorialDay/PresidentsDay)
  was represented in the notebook's headline comparison chart, even though the Interpretation
  section's own prose describes both patterns as the key finding.
- **Redefinition.** "Headline events" (the small subset used for the illustrative side-by-side
  chart, as opposed to the full sortable table) is now defined as: a set representing every
  distinct behavioral pattern found in `event_effect_df`, at least 2 events per pattern, using
  largest-`|event_day_pct|` within a pattern as the tiebreaker — not a top-magnitude-overall
  ranking and not solely the brief's originally-named events. Concrete selection: dip pattern =
  Christmas, Thanksgiving, Easter, SuperBowl (the brief's original 4, kept since all are
  legitimately dip-pattern examples); lift pattern = LaborDay, MemorialDay (the 2 largest-effect
  lift events, added because the brief's set had none). Chart split into two panels (dip vs. lift)
  for visual clarity rather than one panel mixing both.
- **Why this doesn't change the feature decision.** The "Feature implication" in the Interpretation
  cell already selects features (per-`event_name` dummies for the largest-`|event_day|` events,
  plus lead-up/hangover dummies for the closure group) off the full table, not off the headline
  chart's contents — LaborDay was never excluded from the actual feature plan, only from the
  illustrative figure. This is purely a documentation/visualization fix so the figure matches its
  own stated purpose.
- **P2 gate unaffected.** Passed 2026-08-13; the gate's hypothesis-table row for Events cites
  effect sizes from the full table, not the headline chart specifically. `P2-eda.md` was updated
  (Events bullet) to note that the named events are a starting hypothesis and to point future
  headline-event selection at this entry, since the same one-sided-hypothesis risk could recur.

### Events section — event_type aggregate: mean → mean-absolute (2026-08-18)

- **What changed.** The `event_type` bar chart/table (`events_by_type.png`) originally aggregated
  `event_day_pct` with a signed `mean`. Because `National` mixes dip events (Christmas,
  Thanksgiving) with lift events (LaborDay, MemorialDay, PresidentsDay), the signed mean canceled
  to a near-zero, misleading number (~+0.2%), making `National` look like the quietest type when
  it actually contains the largest effects in the dataset in both directions.
- **Fix.** Switched the aggregate to mean **absolute** effect (`s.abs().mean()`), which reports
  typical magnitude regardless of sign. `National` now correctly reads as the largest (5.17%),
  ahead of `Cultural` (3.72%) and `Sporting` (3.44%); `Religious` (0.98%) is genuinely the
  quietest. Table values also switched to `.round(4)` display (4 decimal places) instead of
  formatted percent strings, matching the plain numeric style already used for `type_effect`.
- **Why this doesn't change the feature decision.** The point of this chart/table has always been
  to show that a pooled `event_type` feature is unusable — the mean-absolute version makes that
  case even more directly (large magnitude, direction unknowable) than the misleading near-zero
  signed mean did. The feature implication (per-`event_name` dummies, not per-`event_type`) was
  never derived from this aggregate and is unchanged. Hypothesis-table row #6 in this file and the
  notebook's Events Interpretation cell were both updated to cite the corrected number.
- **P2 gate unaffected.** Passed 2026-08-13; this is a bug fix to an EDA-stage aggregate, not a
  change to scope, invariants, or the gate's feature-decision output.

### Price section — promo inference dropped (2026-08-18)

- **What was there.** The Price section carried a "promo-inference tie-in" (two cells, plus intro
  and interpretation prose): weeks priced >=10% below a series' own trailing baseline were flagged
  as "candidate promo weeks", and their overlap with named-event weeks (45.1%) was compared against
  the all-weeks baseline rate (47.2%) to test whether event dummies would already capture promo
  effects. This was **not** requested by `P2-eda.md` — the brief's Price bullet asks only for
  distribution/dispersion, change frequency, sales response, and relative-vs-raw price. It was
  self-added scope, carried over as a question deferred from the Events section.
- **Why it's dropped.** M5 ships no promotion label, so any promo flag is an inferred construct. A
  thresholded "is this week a promo" indicator is a binarized, strictly lossy version of the
  continuous `relative_price` (price / own trailing baseline) feature the section already computes:
  a GBM can split `relative_price` at whatever cut is actually predictive, per item and in
  interaction with events/SNAP, whereas the 0.90 threshold hardcodes one arbitrary guess. Per
  `CLAUDE.md`'s "does this change what gets built?" test, the check changed no feature decision —
  it came back negative, and the feature it was evaluating is one we now decline to build at all.
- **A validity problem also surfaced, and is moot now.** The flag was described as catching
  "temporary price drops (dip then recover)", but the condition only tested a single week against
  the trailing baseline and never required recovery. Measured directly: 77.1% of flagged weeks are
  still flagged in the series' next observed week, i.e. the majority are sustained repricings (the
  4–15-week rolling baseline simply hasn't caught up yet), not temporary markdowns. Tightening the
  definition to require recovery was considered and rejected — it would have made the label more
  defensible without making it useful, since the continuous feature supersedes it either way.
- **What replaces it.** `relative_price` is kept and now explicitly framed as a *continuous* price
  sensitivity input. Note this does not contradict hypothesis-table row #8: the near-zero Spearman
  correlations reject a **pooled scalar elasticity coefficient**, not price as a model input —
  Spearman measures only pooled monotonic response and cannot see per-item, non-monotonic, or
  price x event/SNAP interaction effects that a GBM can. The price-change-**recency** feature is
  also kept and is unrelated to promo inference; it encodes "how stale is this price", which is
  structurally meaningful given 27.5% of series never reprice.
- **P2 gate unaffected.** Passed 2026-08-13. Removing self-added scope doesn't reopen a brief
  requirement; the gate's hypothesis table has no promo row, row #8 is unchanged and still cites
  `price_change_sales_response.png`, and no figure was deleted (the dropped cells produced printed
  output only).

## P3 — Metrics            (2026-08-18)

### Metric validation gaps found during implementation (2026-08-18)

Recorded while P3 is still open, not at the gate: these are decisions about what the metric does and
what the gate must prove, and the reasoning matters more than the diffs.

- **Aggregated predictions were inflated by the horizon length.** `compute_wrmsse` joined
  predictions to `actuals[["id"] + grouping_cols]` on `id` alone. `actuals` is long format — one row
  per `(id, d)`, so 28 rows per series — which made it a many-to-many join, duplicating every
  prediction row 28× and inflating the summed prediction at levels 2–12 by that factor. Fixed with
  `.drop_duplicates()` on the right-hand lookup (`id → grouping columns` is exactly 1:1: 30,490 ids,
  30,490 distinct triples). **The parquet is not implicated** — `data/processed/sales_long.parquet`
  has zero duplicate `(id, d)` rows and no null prices; the shape was correct and the consumer's
  join key was wrong.
- **An empty weight window scored 0.0 instead of failing.** Weights come from `d_1886`–`d_1913`,
  hardcoded. Any panel outside that range produced no weight rows, and the level score collapsed to
  `(empty × empty).sum() == 0.0` — an apparently perfect score, for *any* forecast. Measured on a
  toy panel: levels 2–12 all returned `0.0` for a forecast wrong by a factor of ~10,000; only
  `Total` responded, and only because its weight is built as a literal one-entry Series. This is the
  most dangerous failure mode the module can have (silent success), so it now raises. `weight_start`
  / `weight_end` were added as parameters, defaulting to the competition window, so tests can supply
  toy day indices instead of being silently no-op'd. **The defaults are unchanged and invariant 1
  still holds** — the window must stay inside the training period.
- **The tests did not test.** A test asserting `level_scores["Store"] == 0.0` passed against the
  deliberately reintroduced fan-out bug, because of the empty-window collapse above. Verified by
  mutation testing (reintroduce the bug, confirm the suite goes red) rather than by the suite being
  green. This motivated validation item 4 and the "a gate item is only met by a check that can fail"
  note now in `P3-metrics.md`. **Lesson worth keeping:** on this phase specifically, "tests pass" is
  not evidence — the metric's failure modes are mostly silent-success, which is exactly what a
  passing test looks like.
- **`np.diff` on `int16` wrapped.** Sales are `int16`; the original implementation differenced and
  squared in that dtype, so any day-over-day change above 181 units overflowed 32,767. Measured on
  `HOBBIES_1_268_CA_1`: naive baseline 285.15 vs. a true 353.70, a 19% understatement of the RMSSE
  *denominator*, which inflates that series' RMSSE. 51 of 30,490 series (0.17%) exceed the
  threshold; largest single-day change in the panel is 645 units. Only the bottom level was exposed
  — `groupby.sum()` promotes aggregates to `int64`. The vectorized rewrite fixes this incidentally
  (pandas promotes to float32); an explicit `float64` cast was added for exact precision and to
  guard the nullable-`Int16` path, where `groupby.diff()` does still return `Int16`.
- **Silent fallbacks replaced with explicit failures.** A series with a weight but no RMSSE was
  scored `0` (treated as a perfect forecast); `compute_mase` filled a missing naive MAE with `1.0`;
  and `compute_naive_baseline_errors` declared a `training_end` parameter it never applied, so a
  kwarg that appeared to enforce the train/holdout boundary did nothing. All three now raise or
  actually filter.
- **Performance: ~1.5–3 hours → 6.5 minutes.** `compute_naive_baseline_errors` looped per series
  doing a full-frame boolean scan each time (O(series × rows); ~21 min at level 12 alone), and four
  `.apply(..., axis=1)` string joins cost ~6.5 min per 46M-row frame. Both vectorized. The remaining
  cost is the repeated 46M-row groupbys across 12 levels, not any Python loop.
- **Toy panel rebuilt to actually satisfy validation item 1.** The original panel used *perfect*
  forecasts, so every expected RMSSE was `0.0` — and `sqrt(0 / x) == 0` for any denominator, so it
  could not detect a mis-scaled naive baseline. It also never exercised weights: `compute_weights`
  and `compute_wrmsse` were never called on it, despite the brief naming weights explicitly. And its
  "hand calculation" cell re-implemented the same logic in numpy and printed it, while the assertion
  compared against a separate hardcoded `{A: 0, B: 0, C: 0}` — so the derivation and the check were
  not connected. Rebuilt with imperfect forecasts giving non-zero hand-computed RMSSE (0.5, 1.0,
  sqrt(1/3)), prices chosen so dollar-sales weights land on 0.2/0.6/0.2, days moved to
  `d_1904`–`d_1913` so they fall inside the weight window, and 10 quantities asserted (naive MSE,
  RMSSE, weights, WRMSSE). Expected values are transcribed literals, not recomputed — a
  re-implementation would reproduce any shared bug. **Verified by mutation:** the panel now rejects
  a 2x-mis-scaled denominator, removal of the leading-zero trimming, and weights computed on units
  instead of dollars.
- **Result.** All-zeros forecast now gives WRMSSE = **5.4465**, all 12 levels finite, aggregate
  levels scoring worse than bottom levels as expected. This satisfies validation item 3 — but note
  it did so *before* these fixes too, which is the point of adding item 4.
- **Gate passed 2026-08-18.** All six items verified in a full notebook run, each derived from a
  computed value rather than a hardcoded `[x]`. Residual soft spot recorded in `P3-metrics.md`: the
  suite catches 1 of 3 injected `compute_wrmsse` bugs (the two survivors are masked by a redundant
  guard and by pandas dtype promotion respectively, so neither is a live hole), making
  `compute_wrmsse` the thinnest coverage in this phase.

<!-- Append phase entries below as gates are passed. -->
