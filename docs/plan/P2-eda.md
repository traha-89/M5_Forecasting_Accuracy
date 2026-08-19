# P2 — Exploratory analysis

**Output:** `notebooks/02_eda.ipynb`, figures in `reports/figures/`
**Effort:** 1–2 sessions
**Gate:** a written findings → feature-hypothesis table, each row citing its evidence

## Objective

EDA here has a job beyond description: **every plot should either kill or justify a candidate feature
for P5.** The gate is the hypothesis table, not the figure count. A beautiful notebook that doesn't
change the P5 feature list has failed this phase.

## Assumes

- P1 passed: `data/processed/sales_long.parquet` exists, pre-release rows dropped.

## Analyses

- [x] Trend and level
- [x] Seasonality
- [x] SNAP
- [x] Events
- [x] Price
- [x] Intermittency
- [x] Lifecycle
- [x] Outliers

**Trend and level.** Total units over time, then split by state, store, category, department. Ask
whether apparent growth is real demand or assortment expansion (more items on sale) — these imply
different features.

**Seasonality.** Day-of-week profile, month-of-year, annual shape. Weekend lift is strong and differs
between FOODS and HOBBIES — quantify per category, not just overall.

**SNAP.** Food-stamp disbursement days should visibly lift FOODS on a state-specific schedule.
Quantify the lift per state. This is one of the highest-value features in the dataset, so it deserves
a proper effect size rather than a glance.

**Events.** Effect size per `event_name` and `event_type` (Sporting, Cultural, National, Religious),
including lead-up and hangover days. Super Bowl, Thanksgiving, Easter, and Christmas behave very
differently from one another — an aggregate "is_event" flag will hide this.

The four events named above are a starting hypothesis, not a prescribed final selection — they
turned out to all share one behavioral pattern (event-day dip, closure/gathering holidays). Once
the full per-event effect table exists, pick the notebook's "headline" comparison set — the small
subset used for the illustrative side-by-side chart — to represent *every* distinct behavioral
pattern found in that table (at least 2 events per pattern), not just the events named here or the
ones with the largest raw effect size. See `DECISIONS.md` ("Events section — headline event
selection") for the concrete example.

**Price.** Distribution and dispersion; sales response to a price change; how often prices move;
whether an item's price *relative to its own history* predicts better than the raw level.

**Intermittency.** Distribution of zero-fraction across series, and the ADI / CV² quadrant
classification (smooth, erratic, intermittent, lumpy):

- ADI = average inter-demand interval = periods / number of non-zero periods
- CV² = squared coefficient of variation of non-zero demand
- Standard cutoffs: ADI = 1.32, CV² = 0.49

Persist the per-series quadrant label to **`data/processed/series_segments.parquet`**
(`id`, `adi` `float32`, `cv2` `float32`, `quadrant` `category`) — P6 and P7 reuse it for the segment
breakdown that decides whether to route different models to different series.

**Lifecycle.** Release dates over time; series that die mid-panel; the practical shape of "new item,
no history".

**Outliers.** Extreme single-day spikes. Distinguish genuine promotional demand (keep — it is signal)
from implausible values. RMSSE squares errors, so a handful of spikes can dominate a series' score.
Decide the handling policy here and record it in `DECISIONS.md`; note that clipping spikes will
improve the metric while making the forecast less useful operationally.

## Output

A table in the notebook, and copied to `DECISIONS.md`, of the form:

| Finding | Evidence | Proposed feature | Verdict |
|---|---|---|---|
| FOODS lifts ~15% on TX SNAP days | fig `snap_by_state.png` | `snap_<state>` matched to series state | adopt |

Target at least 10 rows. A "reject" verdict is a valid and useful outcome.

## Gate

- [x] Hypothesis table complete, ≥10 rows, each citing a figure
- [x] ADI/CV² quadrant labels persisted to `data/processed/series_segments.parquet`
- [x] Outlier handling policy decided and recorded in `DECISIONS.md`
- [x] Figures saved to `reports/figures/`

**Passed 2026-08-13** — see `DECISIONS.md` ("P2 — Exploratory analysis" section, outlier policy
decision + hypothesis table entries).

## Invariants

- **All EDA is computed on `d_1`–`d_1913` only** (invariant 1). Plotting the holdout period is a
  subtle form of peeking — it shapes feature choices around data we have promised not to use.
- Aggregate carefully on 46M+ rows; use `groupby` on the parquet rather than materialising wide
  frames (invariant 3).

Full list: [README](README.md#invariants).
