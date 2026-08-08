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

## Gate

- [ ] Hand-computed toy panel matches implementation
- [ ] Unit tests pass
- [ ] All 12 levels produced, 42,840 series total
- [ ] All-zeros forecast gives a finite WRMSSE
- [ ] `src/metrics.py` importable; notebook imports rather than redefines

## Invariants

- Weights use the last 28 days of the **training** period (`d_1886`–`d_1913` for the final fold),
  never the holdout (invariant 1).
- The metric function takes actuals as an argument; it must not read the holdout itself.

Full list: [README](README.md#invariants).
