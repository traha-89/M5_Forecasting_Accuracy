# P2 EDA — inventory/demand-planning practitioner review

**Purpose:** external sanity check on the P2 findings from a demand-planning practitioner's
perspective, ahead of closing out the phase. Generated via the `/inventory-demand-planning` skill
reviewing `notebooks/02_eda.ipynb`'s outputs. Not a gate requirement — a second opinion to weigh
before P2 is considered fully wound up, and to carry forward into P9 (replenishment) if that phase
is later approved.

**Date:** 2026-08-13

---

**Portfolio shape is the single most important finding, and it changes everything downstream.**
91% of series fall in the sparse ADI/CV² quadrants (72.7% intermittent, 18.4% lumpy) — only 6.1%
are smooth. That means the textbook `SS = Z × σ_d × √(LT)` safety-stock formula is invalid for
nine-tenths of this assortment out of the gate; it assumes roughly-normal, stationary demand, and
lumpy/intermittent series violate that badly. This portfolio needs Croston's/SBA forecasting and
bootstrapped safety-stock distributions as the *default*, not the exception — flagging this as the
top risk if P9's replenishment policy leans on a single quantile-based reorder point without
segmenting by this label first. The `series_segments.parquet` output is exactly the right artifact
to gate that segmentation on.

**Trend finding is a real red flag for buy planning, not just an EDA curiosity.** +81.3% raw trend
sounds like healthy demand growth; the assortment breakdown shows units-per-active-series actually
*fell* 36.7% while catalog size grew 153%. In planner terms: this looks like SKU proliferation
diluting per-item velocity, not organic same-item demand growth. An ABC cut layered on top of this
would be needed before trusting any aggregate trend signal in a forecast — a portfolio that's
growing in count but shrinking in per-item pull is a classic setup for creeping excess inventory if
buy quantities are set off the wrong number.

**SNAP and Events read like clean, usable causal signals** — better than most retail datasets,
actually, because they're deterministic and known in advance rather than inferred. The
state-specific SNAP lift (WI +2.5, TX +2.0, CA +0.2 pct pts) is exactly the kind of recurring,
calendar-known demand pulse that would feed DC-level replenishment timing for FOODS, state by
state — not unlike a recurring TPR with no forecast uncertainty on *timing*, only magnitude. Same
for the closure-holiday pattern (Thanksgiving +15% lead-up, then a dip on the day): that's a
textbook forward-buy/pre-holiday pull-forward, and the notebook caught the underlying mechanism
correctly rather than just flagging "holiday = anomaly."

**Price section is the one place to caution against over-applying a standard promo playbook.**
Standard retail-planning frameworks assume frequent, flagged promotions with lift/cannibalization/
forward-buy dynamics. This dataset doesn't have that texture — prices change in 0.7% of weeks,
27.5% of series never change price at all. The notebook's conclusion (reject a pooled
price-elasticity feature, the correlation is noise at -0.03/+0.02) is the right call. If a real
promo calendar ever becomes available for this business, cannibalization and post-promo dip
modeling would matter far more than they do against this specific dataset's price history.

**Outlier policy — agree with "don't clip," and here's the practitioner reason why:** clipping
promotional spikes is one of the most common self-inflicted forecasting errors in practice,
because it teaches the model that big demand days don't happen, right up until one does and you
stock out. 274 extreme days out of 46M rows, mostly plausible bulk-purchase magnitudes, 43% tied
to a known driver already — this is exactly the profile where keeping the signal and building
explanatory features (which the notebook already did — SNAP/event/price-drop flags) is the correct
move over target manipulation.

**One gap for P9 specifically:** the EDA gives quadrant labels (predictability) but not ABC
(value) tiers — no revenue/margin cut appears anywhere in this notebook. Before building
reorder-point logic, that should be cross-tabulated (AX vs. CZ), since the safety-stock
service-level target should differ by cell, not just by demand pattern.
