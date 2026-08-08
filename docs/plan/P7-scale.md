# P7 — Scale out, evaluate, select

**Output:** `notebooks/07_model_full.ipynb` + a model card
**Effort:** 2–3 sessions, plus overnight runs
**Gate:** holdout WRMSSE reported **once**, with a full comparison table and per-level breakdown

## Objective

Run the pilot's pipeline across all 10 stores and all 30,490 series, select the final model on
cross-validation, then score it once on the true holdout.

## Assumes

- P6 passed: pipeline validated on CA_1, runtimes known, horizon strategy decided.
- P5 passed: all ten per-store feature partitions buildable.

## This is the phase where the discipline matters

The holdout actuals for `d_1914`–`d_1941` are sitting in `sales_train_evaluation.csv`, the same file
we train from. Scoring against them repeatedly and picking the best result is not evaluation — it is
selection on the test set, and the reported number becomes fiction.

**The order is fixed and not negotiable:**

1. Train every configuration across all 10 store partitions; run StatsForecast over all 30,490 series.
2. Score everything on the **rolling-origin CV folds**. Pick the winning model and any ensemble
   blend weights **here**, from CV alone.
3. *Then*, once, score the selected model on `d_1914`–`d_1941`, full 42,840-series WRMSSE.
4. Whatever step 3 returns is what gets reported. Do not go back and re-select.

If step 3 disappoints, that is a finding to write up, not a reason to re-run selection.

## Execution notes

Expect roughly 2–4 hours per gradient-boosted configuration on 8 cores (confirm against the P6
extrapolation). This phase is scheduled around overnight runs rather than interactive iteration —
another reason design questions were settled in P6.

Train store partitions sequentially, persisting each model to disk, so a failure at store 8 does not
lose stores 1–7.

## Diagnostics

Break results down along four axes — this is where the interesting findings live:

- **Per aggregation level** (all 12). Catches models that are good at the bottom but biased in
  aggregate, or vice versa.
- **Per store** and **per category.** Is one store dragging the average?
- **Per ADI/CV² quadrant** (from P2). The key question: do the gradient-boosted models win on dense
  series while the Croston family wins on genuinely sparse ones?
- **Calibration.** Predicted vs actual totals at each level, plus the bias number. Count models
  systematically under-forecast; check whether this one does.

## The segmentation decision

**Only route different models to different segments if the quadrant breakdown supports it.** If
StatsForecast beats the gradient-boosted models on the intermittent/lumpy quadrants by a clear margin,
a routed ensemble is justified by evidence. If it doesn't, take the single best model and skip the
complexity — a routing rule fitted to noise is worse than no routing.

Whatever is decided, blend or routing weights are fit on **CV folds**, never on the holdout.

## Output — the model card

Record in `DECISIONS.md` and the notebook:

- Chosen approach and why
- Holdout WRMSSE, plus per-level breakdown
- Comparison against all five P4 baselines and all four P6 candidates
- Known weaknesses (which segments/levels it handles poorly)
- Training runtime and resource profile
- Anything that would be tried next given more time

## Gate

- [ ] All configurations trained across 10 stores; StatsForecast over all 30,490 series
- [ ] Models persisted to `models/full/<model>_<store>.<ext>`
- [ ] `reports/full_scores.csv` written and committed
- [ ] Model and blend weights selected on **CV only**, before the holdout is touched
- [ ] Holdout WRMSSE computed exactly once, over the full 42,840 series
- [ ] Breakdown by level, store, category, and ADI/CV² quadrant
- [ ] Segmentation decision made on evidence and recorded
- [ ] Model card written

## Invariants

- Invariant 1 is the whole point of this phase — see the fixed order above.
- Per-store partitions; do not load all ten feature sets at once (invariant 3).
- Encodings and blend weights fit on training folds only (invariant 4).

Full list: [README](README.md#invariants).
