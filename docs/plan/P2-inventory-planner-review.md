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

---

## Re-review (2026-08-19)

The notebook has been through several rounds of interpretation trims and figure fixes since the
above was written. Every number this review cites — portfolio quadrant split (72.7%/18.4%/6.1%),
trend (+81.3% raw / -36.7% per-active-series / +152.9% catalog), SNAP lift (WI +2.5, TX +2.0,
CA +0.2), price change rate (0.7%/27.5%), and outlier policy (274 extreme days, 43.2% explained)
— is unchanged. The findings and recommendations above still stand. Two things are worth
amending, not because anything was wrong, but because the notebook now says more than it did:

**Softening my own Croston's/SBA framing — one word swap that matters.** The notebook's
Intermittency section now states the quadrant label should drive *loss/objective choice within
the project's already-fixed GBM family* (Tweedie/Poisson for `intermittent`, something beyond a
single Tweedie fit for `lumpy`) rather than a switch to Croston's/SBA. That's a legitimate call,
not a gap — this project committed to LightGBM/XGBoost from the start (per `CLAUDE.md`'s model
shortlist), and a well-tuned Tweedie-objective GBM with SNAP/event/price features already built in
is a reasonable, arguably stronger, substitute for Croston's on lumpy retail demand, which
classical intermittent-demand methods can't take side-information from at all. I'd revise my
original wording ("needs Croston's/SBA... as the default") — that's true only if you're picking a
*forecasting method* from a blank slate, which this project isn't.
**But this doesn't touch the actual risk I was flagging.** My concern was never really about which
point-forecast algorithm to use — it's that **safety stock and reorder-point sizing for P9 still
can't use the textbook normal-distribution formula for 91% of this portfolio**, no matter how good
the point forecast is. A Tweedie-GBM forecast for a lumpy series is still a forecast of a lumpy,
non-normal distribution; P9 still needs a bootstrapped or simulation-based safety-stock
calculation off that distribution, not `SS = Z × σ_d × √(LT)`. That risk is unchanged and still the
top item to carry forward.

**One reinforcing cross-reference, not in the original review.** The Outliers section now reports
the driver breakdown for the 43.2% of spikes with a known cause: SNAP alone explains 378k of the
436k explained spike-days (86.7%) — well ahead of events (76.8k) and price drops (18.5k). That's a
second, independent confirmation of this review's SNAP paragraph: state-specific SNAP timing isn't
just a clean baseline-demand signal, it's also the dominant explanation for *why* individual days
spike, which strengthens the case for SNAP-aware DC-level replenishment timing specifically for
FOODS, state by state.

**The ABC/value-tier gap for P9 is still open** — nothing in this round of edits added a
revenue/margin cut. Unchanged recommendation: cross-tabulate quadrant × ABC before building
reorder-point logic.
