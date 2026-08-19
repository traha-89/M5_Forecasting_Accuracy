# P3 — Metrics

**Output:** `notebooks/03_metrics.ipynb` + `src/metrics.py`
**Effort:** 1 session
**Gate:** WRMSSE reproduces a hand-computed value on a toy panel; unit tests pass

## Objective

Implement the competition metric and its supporting diagnostics **before any model is built.** A
subtly wrong WRMSSE silently invalidates every comparison downstream, and it is far cheaper to
validate against a hand-computed toy example now than to debug from a suspicious leaderboard later.

This phase writes `src/metrics.py` as well as the notebook — five later notebooks import it, and
copy-pasting the metric between them will cause drift.

## Assumes

- P1 passed: long-format parquet exists with prices joined.

## WRMSSE

For each series:

```
RMSSE = sqrt( mean_h( (y - ŷ)² ) / mean_t( (y_t - y_{t-1})² ) )
```

Numerator: mean squared forecast error over the 28-day horizon.
Denominator: mean squared one-step naive error over the training period.

Two details that are easy to miss and both change the number materially:

1. **The denominator is computed only from the series' first non-zero sale onward.** Including the
   leading pre-release zeros understates the scale and flatters the model.
2. **Weights are each series' share of dollar sales (units × price) over the final 28 days of the
   training period**, normalised within each aggregation level.

## The 12 aggregation levels

WRMSSE is computed across all twelve, totalling **42,840 series**:

| Level | Grouping | Series |
|---|---|---|
| 1 | Total | 1 |
| 2 | State | 3 |
| 3 | Store | 10 |
| 4 | Category | 3 |
| 5 | Department | 7 |
| 6 | State × Category | 9 |
| 7 | State × Department | 21 |
| 8 | Store × Category | 30 |
| 9 | Store × Department | 70 |
| 10 | Item | 3,049 |
| 11 | Item × State | 9,147 |
| 12 | Item × Store | 30,490 |
| | **Total** | **42,840** |

WRMSSE is the **unweighted mean of the twelve weighted level scores**. A model can therefore be
excellent at item–store level and still score badly if its aggregates are biased — worth knowing
early, and the reason per-level output is a required diagnostic.

## Supporting metrics

| Metric | Role | Why it earns a slot |
|---|---|---|
| WRMSSE | **primary** | The competition metric. All decisions made on this. |
| RMSSE, unweighted | diagnostic | Bottom level, equal weight. Separates "good at big items" from "good everywhere". |
| MAE / WMAE | diagnostic | Absolute-error view, far less spike-dominated. Disagreement with RMSSE localises outlier sensitivity. |
| MASE | diagnostic | Same scaling idea in absolute terms; widely legible cross-check. |
| Total bias % | guardrail | Count models systematically under-forecast. A blunt bias number catches it immediately. |
| Per-level WRMSSE | diagnostic | The twelve components separately. The most useful debugging output in the project. |

**Deliberately excluded: sMAPE and MAPE.** Both divide by actuals, and with ~68% zeros they are
undefined or explosive across most of this dataset. Do not add them.

## Validation

The gate is validation, not implementation. Three independent checks:

1. **Toy panel.** Construct a small panel (say 3 series × 10 training days × 3 horizon days) with
   values chosen so RMSSE and the weights can be computed by hand. Assert the implementation matches
   to floating-point tolerance. Show the hand computation in the notebook.
2. **Unit tests** in `src/metrics.py` or a `tests/` file: zero-error forecast → RMSSE 0; naive forecast
   on a random-walk series → RMSSE ≈ 1; weights sum to 1 within each level; 42,840 series produced.
3. **Degenerate baseline.** An all-zeros forecast should produce a finite, sane WRMSSE. If it produces
   `inf` or `nan`, the denominator handling of the leading-zeros rule is wrong.
4. **Aggregation check, with a non-zero forecast.** Take a small panel with at least two series
   sharing one grouping value, forecast a known non-zero constant, and assert the aggregated
   prediction at a grouped level equals the hand-summed total. Checks 1–3 cannot catch an
   aggregation error: the toy panel is too small, and an all-zeros forecast is precisely the input
   under which a multiplicative aggregation bug is invisible (`0 × n == 0`). Two conditions the
   check must satisfy or it passes vacuously:
   - **Horizon longer than one day.** A one-day horizon makes any fan-out factor 1.
   - **Days inside the weight window.** Weights come from the final 28 training days; a toy panel
     outside that range yields no weight rows, and every non-total level collapses to `0.0`
     regardless of the forecast. Assert the panel actually reaches the window.

   See `DECISIONS.md` ("P3 — metric validation gaps found during implementation") for the concrete
   bugs that motivated this item.

## Gate

- [x] Hand-computed toy panel matches implementation
- [x] Unit tests pass
- [x] All 12 levels produced, 42,840 series total
- [x] All-zeros forecast gives a finite WRMSSE
- [x] Non-zero forecast reproduces a hand-summed aggregate at a grouped level (validation 4)
- [x] `src/metrics.py` importable; notebook imports rather than redefines

A gate item is only met by a check that can fail. Before ticking any box here, confirm the
corresponding test actually goes red when the behaviour it guards is broken — several tests written
during P3 passed against a deliberately reintroduced bug.

**Passed 2026-08-18** — see `DECISIONS.md` ("P3 — Metrics", metric validation gaps found during
implementation). Evidence: all six items derived from computed values in a full run of
`03_metrics.ipynb` (no hardcoded `[x]`); 18 unit tests pass; all-zeros WRMSSE = 5.4465 with all 12
levels finite; 42,840 series confirmed present in the weight window, not just countable in the raw
data. The toy panel and the aggregation guard were each mutation-tested — the toy panel rejects a
2x-mis-scaled denominator, removal of the leading-zero trimming, and unit-based weights; the
aggregation test rejects the fan-out join.

Known residual: the unit suite catches 1 of 3 injected `compute_wrmsse` bugs. Both survivors were
traced — one is masked by a redundant guard, the other by pandas dtype promotion — so neither is a
live hole, but `compute_wrmsse` coverage is the thinnest part of this phase and is worth
strengthening if P6/P7 comparisons ever look suspicious.

## Invariants

- Weights use the last 28 days of the **training** period (`d_1886`–`d_1913` for the final fold),
  never the holdout (invariant 1).
- The metric function takes actuals as an argument; it must not read the holdout itself.

Full list: [README](README.md#invariants).
