# P9 — Replenishment policy

**Status: proposed, not yet approved into the phase sequence.** P0–P8 remain the committed plan;
this brief is parked for future review, not an active gate. Do not start building against it
without an explicit go-ahead.

**Output:** `notebooks/09_replenishment.ipynb`, `src/replenishment.py`,
`reports/replenishment_comparison.csv`
**Effort:** 1–2 sessions
**Gate:** measurable improvement over the naive baseline (stockout rate or excess inventory) on
folds 1–3, no peeking at holdout/forecast

## Objective

Extend the pipeline with an inventory replenishment layer on top of the forecast: given demand
*uncertainty* (not just a point forecast), compute a reorder point per series and compare it
against a naive baseline.

Deliberately narrow for this pass: only the core reorder-point-from-quantile calculation. Two more
advanced pieces from the source paper (Brunaud et al., extended by a Zalando paper on an (R,s,Q)
policy) — a kick-start quantity for new items and a lifecycle cutoff for aging items — are
out of scope for this phase. See "Parked — P10 candidate" below.

## Dependency to check first — confirmed, not assumed

This phase needs a demand **distribution**, not just a point forecast. Checked directly against
`P6-pilot.md` and `P7-scale.md` rather than assumed:

**Only a point forecast exists.** All four P6 candidates — LightGBM Tweedie, LightGBM Poisson,
XGBoost (`count:poisson`), and the Nixtla StatsForecast ensemble (Croston/ADIDA/IMAPA/TSB/
SeasonalNaive/DynamicOptimizedTheta) — are configured as point-forecast objectives; none has a
quantile objective or interval configured, and P7 reuses P6's model set unchanged. This confirms
the note already on record in `DECISIONS.md` (Pre-P0, "Still open"): *"Prediction intervals — out
of scope. M5 had a companion Uncertainty track using pinball loss across nine quantiles; LightGBM
can produce these with quantile objectives at ~9× training cost."* P9 reopens that closed
question and must resolve it before anything else here can be built.

Two options, to be decided **when this phase actually kicks off**, not now:
- **(a) Re-run the P7 winning model with a quantile objective** (e.g. LightGBM `quantile` at
  q25/q50/q75) as a small addendum. Higher fidelity, but the ~9× per-quantile training cost noted
  above is real — three quantiles is closer to 3× the winning model's training time, not 9×, since
  only the needed quantiles need refitting, but confirm this against actual P7 runtimes before
  committing.
- **(b) Approximate a distribution from each series' historical residual std dev** (CV-fold
  residuals, already available from `reports/full_scores.csv`-adjacent per-series errors, if
  persisted — check this exists before assuming it). Weaker (assumes a fixed-shape error
  distribution per series, ignores heteroskedasticity across the ADI/CV² quadrants), but requires
  no retraining.

Whichever is chosen, record the choice and the evidence for it in `DECISIONS.md` **at that time**
— not pre-decided here, since the right answer depends on what P7 actually produced and what
compute budget is available then.

## Assumes (once the dependency above is resolved)

- `reports/full_scores.csv` and `models/full/<model>_<store>.<ext>` from P7
- A per-series quantile forecast at minimum q75 over the forecast horizon (from whichever option
  above is chosen)
- `data/processed/series_segments.parquet` from P2 — useful for checking the policy behaves
  sensibly differently on smooth vs. intermittent/lumpy series, the same quadrant breakdown P7
  already uses for model diagnostics

## Build

- `09_replenishment.ipynb` implementing, per series:
  - Reorder point = q75 forecast over a lead time. **M5 has no explicit lead time** — state a
    placeholder assumption in `DECISIONS.md` when this phase starts, don't invent one here.
  - A naive `(s,S)` baseline using the classical z-score × σ × √(lead time) safety stock formula,
    for comparison.
- `src/replenishment.py` — the policy function(s), following the same "single definition used
  everywhere" convention as `src/metrics.py` (P3) and `src/features.py` (P5): one module imported
  by the notebook, not redefined inline.
- `reports/replenishment_comparison.csv` — per series/fold: simulated stockouts avoided, excess
  stock, vs. the naive baseline.

## Parked — P10 candidate

Not built in this phase. Noted here so the scope cut is explicit rather than silently dropped:

- **Kick-start quantity for new items** — an initial stocking quantity for series with little or
  no sales history (see P2's Lifecycle section: 1.36% of series have under a year of history,
  though 0% have under 28 days).
- **Lifecycle cutoff for aging items** — a rule for winding down replenishment on series trending
  toward discontinuation (see P2's Lifecycle section: 0.6% of series already show a >=180-day
  dormant tail even mid-panel).

## Gate

- [ ] Dependency resolved: quantile/distribution source decided and recorded in `DECISIONS.md`
- [ ] `src/replenishment.py` implements the (R,s,Q)-style policy and the naive baseline
- [ ] `reports/replenishment_comparison.csv` written and committed
- [ ] Policy shows measurable improvement over the naive baseline (stockout rate or excess
      inventory) on folds 1–3
- [ ] Per-ADI/CV²-quadrant breakdown checked — does the policy behave sensibly differently on
      smooth vs. intermittent/lumpy series, or does it need quadrant-specific tuning
- [ ] Lead-time placeholder assumption recorded in `DECISIONS.md`

## Invariants

Same invariants as the rest of the plan apply here, in particular:

- **No peeking at `d_1914`–`d_1941` (holdout) or `d_1942`–`d_1969` (forecast) for policy tuning**
  (invariant 1) — the policy is fit and compared on folds 1–3 exactly like the models it sits on
  top of.
- **Chronological evaluation only** (invariant 5) — stockout/excess-stock simulation runs forward
  through each fold, not on a random split.

Full list: [README](README.md#invariants).
