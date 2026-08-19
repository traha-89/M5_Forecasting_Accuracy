"""Unit tests for src/metrics.py."""

import numpy as np
import pandas as pd
import pytest
from src import metrics


class TestComputeNaiveBaselineErrors:
    """Test compute_naive_baseline_errors."""

    def test_constant_series_zero_error(self):
        """Constant series should have zero MSE (no variance)."""
        actuals = pd.DataFrame({
            'id': ['A'] * 5,
            'd': [1, 2, 3, 4, 5],
            'sales': [10, 10, 10, 10, 10],
        })
        result = metrics.compute_naive_baseline_errors(actuals)
        assert result['A'] == 0.0

    def test_linear_series_unit_error(self):
        """Linear series with unit steps should have MSE = 1."""
        actuals = pd.DataFrame({
            'id': ['A'] * 5,
            'd': [1, 2, 3, 4, 5],
            'sales': [1, 2, 3, 4, 5],
        })
        result = metrics.compute_naive_baseline_errors(actuals)
        assert result['A'] == 1.0

    def test_all_zeros(self):
        """All-zero series should have MSE = 0."""
        actuals = pd.DataFrame({
            'id': ['A'] * 5,
            'd': [1, 2, 3, 4, 5],
            'sales': [0, 0, 0, 0, 0],
        })
        result = metrics.compute_naive_baseline_errors(actuals)
        assert result['A'] == 0.0

    def test_sparse_series_starts_from_first_nonzero(self):
        """Diffs should be computed only from first non-zero sale onward."""
        actuals = pd.DataFrame({
            'id': ['A'] * 7,
            'd': [1, 2, 3, 4, 5, 6, 7],
            'sales': [0, 0, 1, 0, 1, 0, 1],  # Starts at day 3
        })
        result = metrics.compute_naive_baseline_errors(actuals)
        # From day 3: [1, 0, 1, 0, 1]
        # Diffs: [-1, 1, -1, 1]
        # MSE = mean(1, 1, 1, 1) = 1.0
        assert result['A'] == 1.0

    def test_int16_sales_do_not_overflow(self):
        """Large day-over-day swings must not wrap in int16.

        `sales` is int16 on the real panel. A change of 300 units squares to 90,000,
        well past int16's 32,767 limit, so differencing in the native dtype wraps and
        understates the naive baseline — the RMSSE denominator — inflating every RMSSE
        built on it. 51 of the 30,490 real series exceed the 181-unit threshold.
        """
        vals = np.array([0, 300, 0, 300, 0], dtype=np.int16)
        actuals = pd.DataFrame({
            'id': ['A'] * 5,
            'd': [1, 2, 3, 4, 5],
            'sales': vals,
        })
        assert actuals['sales'].dtype == np.int16, "test must exercise the int16 path"

        result = metrics.compute_naive_baseline_errors(actuals)

        # Trimmed to the first non-zero day: [300, 0, 300, 0]
        # diffs [-300, 300, -300] -> squares all 90,000 -> mean 90,000
        assert result['A'] == pytest.approx(90000.0), (
            f"got {result['A']}; a value below 90,000 means the squared differences "
            "wrapped in int16 instead of being computed in float64"
        )

    def test_multiple_series(self):
        """Test multiple series simultaneously."""
        actuals = pd.DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'd': [1, 2, 1, 2],
            'sales': [5, 5, 1, 2],
        })
        result = metrics.compute_naive_baseline_errors(actuals)
        assert result['A'] == 0.0
        assert result['B'] == 1.0


class TestComputeRMSSE:
    """Test compute_rmsse."""

    def test_perfect_forecast_constant_series(self):
        """Perfect forecast on constant series should give RMSSE = 0."""
        train = pd.DataFrame({
            'id': ['A'] * 5,
            'd': [1, 2, 3, 4, 5],
            'sales': [10, 10, 10, 10, 10],
        })
        actuals = pd.DataFrame({
            'id': ['A'] * 3,
            'd': [6, 7, 8],
            'sales': [10, 10, 10],
        })
        preds = pd.DataFrame({
            'id': ['A'] * 3,
            'd': [6, 7, 8],
            'pred': [10, 10, 10],
        })
        result = metrics.compute_rmsse(actuals, preds, train)
        assert result['A'] == 0.0

    def test_naive_forecast_linear_series(self):
        """Naive forecast on perfect random walk should give RMSSE ~ 1."""
        train = pd.DataFrame({
            'id': ['A'] * 5,
            'd': [1, 2, 3, 4, 5],
            'sales': [1, 2, 3, 4, 5],
        })
        # Naive forecast: repeat last value
        actuals = pd.DataFrame({
            'id': ['A'] * 3,
            'd': [6, 7, 8],
            'sales': [6, 7, 8],
        })
        preds = pd.DataFrame({
            'id': ['A'] * 3,
            'd': [6, 7, 8],
            'pred': [5, 5, 5],  # Last training value
        })
        result = metrics.compute_rmsse(actuals, preds, train)
        # MSE = mean((1, 2, 3)^2) = mean(1, 4, 9) = 14/3
        # Naive MSE = 1.0
        # RMSSE = sqrt(14/3) ~ 2.16
        assert result['A'] > 1.5  # Check it's not 0

    def test_multiple_series_separate_scaling(self):
        """Each series should be scaled independently."""
        train = pd.DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'd': [1, 2, 1, 2],
            'sales': [1, 2, 100, 101],  # Different scales
        })
        actuals = pd.DataFrame({
            'id': ['A', 'B'],
            'd': [3, 3],
            'sales': [3, 102],
        })
        preds = pd.DataFrame({
            'id': ['A', 'B'],
            'd': [3, 3],
            'pred': [3, 102],  # Both perfect
        })
        result = metrics.compute_rmsse(actuals, preds, train)
        # Both should be 0 despite different scales
        assert result['A'] == 0.0
        assert result['B'] == 0.0


class TestComputeWeights:
    """Test compute_weights."""

    def test_weights_sum_to_one(self):
        """Weights should sum to 1."""
        sales = pd.DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'd': [1886, 1913, 1886, 1913],
            'sales': [10, 20, 30, 40],
            'sell_price': [1.0, 1.0, 2.0, 2.0],
        })
        weights = metrics.compute_weights(sales)
        assert np.isclose(weights.sum(), 1.0)

    def test_weights_dollar_basis(self):
        """Weights should be based on dollar sales."""
        sales = pd.DataFrame({
            'id': ['A', 'B'],
            'd': [1890, 1890],
            'sales': [10, 10],  # Same units
            'sell_price': [1.0, 2.0],  # Different prices
        })
        weights = metrics.compute_weights(
            sales,
            weight_start=1890,
            weight_end=1890
        )
        # A: 10 units × $1 = $10
        # B: 10 units × $2 = $20
        # Total: $30
        # w_A = 10/30 = 1/3, w_B = 20/30 = 2/3
        assert np.isclose(weights['A'], 1/3)
        assert np.isclose(weights['B'], 2/3)

    def test_weights_respect_date_range(self):
        """Weights should only use specified date range."""
        sales = pd.DataFrame({
            'id': ['A'] * 4,
            'd': [1, 1886, 1890, 1913],
            'sales': [100, 10, 10, 10],  # Large early value, small later
            'sell_price': [1.0, 1.0, 1.0, 1.0],
        })
        weights = metrics.compute_weights(
            sales,
            weight_start=1886,
            weight_end=1913
        )
        # Should ignore the d=1 row
        # Total in range: 10 + 10 + 10 = 30
        assert np.isclose(weights['A'], 1.0)


class TestComputeAllZerosForecast:
    """Test that all-zeros forecast produces finite WRMSSE."""

    def test_zero_forecast_finite_result_total_only(self):
        """All-zeros forecast should not produce inf/nan (total level only)."""
        # Days must sit inside the weight window (1886-1913). Outside it there are no
        # weight rows at all, and compute_wrmsse now raises rather than silently
        # scoring 0.0 — which is what this test used to do.
        train = pd.DataFrame({
            'id': ['A', 'A', 'B', 'B', 'C', 'C'],
            'd': [1886, 1887, 1886, 1887, 1886, 1887],
            'sales': [5, 5, 10, 11, 0, 0],
            'sell_price': [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        })
        actuals = pd.DataFrame({
            'id': ['A', 'B', 'C'],
            'd': [1888, 1888, 1888],
            'sales': [5, 12, 0],
            'sell_price': [1.0, 2.0, 3.0],
        })
        preds = pd.DataFrame({
            'id': ['A', 'B', 'C'],
            'd': [1888, 1888, 1888],
            'pred': [0, 0, 0],  # All zeros
        })

        # Only test with total level (no grouping columns)
        wrmsse, level_scores = metrics.compute_wrmsse(
            actuals, preds, train,
            level_grouping=[[]]  # Only total level
        )

        # Should be finite
        assert np.isfinite(wrmsse), f"WRMSSE is {wrmsse}, not finite"
        assert not np.isnan(wrmsse), "WRMSSE is NaN"
        assert not np.isinf(wrmsse), "WRMSSE is inf"
        # Should be > 0 since forecast is wrong
        assert wrmsse > 0, "WRMSSE should be > 0 for wrong forecast"


class TestFanOutBugFix:
    """Guards the id -> grouping-column lookup join in compute_wrmsse.

    `actuals` is long format: one row per (id, d). Joining predictions to it on `id`
    alone is many-to-many, so every prediction row is duplicated once per horizon day
    and the aggregated sum is inflated by the horizon length. An all-zeros forecast
    cannot detect this (0 * n == 0), so these tests use non-zero forecasts.

    Note the day indices: they sit inside WEIGHT_PERIOD, because a panel outside the
    weight window produces no weights and every non-total level collapses to 0.0,
    which would make these assertions vacuously true.
    """

    # Days chosen to fall inside the default weight window (1886-1913).
    D_TRAIN = [1886, 1887]
    D_HOLD = [1888, 1889, 1890]

    def _panel(self, pred_value_item1, pred_value_item2):
        n_tr, n_ho = len(self.D_TRAIN), len(self.D_HOLD)
        train = pd.DataFrame({
            'id': ['item1', 'item2'] * n_tr,
            'd': [d for d in self.D_TRAIN for _ in range(2)],
            # non-constant, so the naive denominator is > 0 and RMSSE is finite
            'sales': [100, 50, 130, 65],
            'sell_price': [1.0] * (2 * n_tr),
            'store_id': ['A'] * (2 * n_tr),
            'state_id': ['CA'] * (2 * n_tr),
            'cat_id': ['CAT1'] * (2 * n_tr),
            'dept_id': ['DEPT1'] * (2 * n_tr),
            'item_id': ['item1', 'item2'] * n_tr,
        })
        actuals = pd.DataFrame({
            'id': ['item1', 'item2'] * n_ho,
            'd': [d for d in self.D_HOLD for _ in range(2)],
            'sales': [100, 50] * n_ho,
            'sell_price': [1.0] * (2 * n_ho),
            'store_id': ['A'] * (2 * n_ho),
            'state_id': ['CA'] * (2 * n_ho),
            'cat_id': ['CAT1'] * (2 * n_ho),
            'dept_id': ['DEPT1'] * (2 * n_ho),
            'item_id': ['item1', 'item2'] * n_ho,
        })
        preds = pd.DataFrame({
            'id': ['item1', 'item2'] * n_ho,
            'd': [d for d in self.D_HOLD for _ in range(2)],
            'pred': [pred_value_item1, pred_value_item2] * n_ho,
        })
        return train, actuals, preds

    def test_store_level_perfect_forecast_scores_zero(self):
        """Exact forecast must score 0 at Store level.

        Under the fan-out bug the store aggregate becomes 3 x 150 = 450 against an
        actual of 150, so this is non-zero (in fact inf).
        """
        train, actuals, preds = self._panel(100, 50)
        _, level_scores = metrics.compute_wrmsse(actuals, preds, train)
        assert level_scores['Store'] == pytest.approx(0.0), (
            f"perfect forecast scored {level_scores['Store']} at Store level; "
            "a non-zero value means predictions were inflated by the fan-out join"
        )

    def test_grouped_levels_all_zero_for_perfect_forecast(self):
        """Every level, not just Store, must score 0 for an exact forecast."""
        train, actuals, preds = self._panel(100, 50)
        _, level_scores = metrics.compute_wrmsse(actuals, preds, train)
        nonzero = {k: float(v) for k, v in level_scores.items() if v != 0.0}
        assert not nonzero, f"levels scored non-zero on a perfect forecast: {nonzero}"

    def test_weight_window_actually_populated(self):
        """Meta-test: the panel must reach the weight window.

        If it does not, weights are empty and the assertions above pass vacuously.
        This test fails loudly rather than letting the suite go quietly green.
        """
        train, _, _ = self._panel(100, 50)
        in_window = train[
            (train['d'] >= metrics.WEIGHT_PERIOD_START)
            & (train['d'] <= metrics.WEIGHT_PERIOD_END)
        ]
        assert not in_window.empty, (
            "test panel has no rows in the weight window; the fan-out assertions "
            "above would be vacuous"
        )

    def test_empty_weight_window_raises(self):
        """A panel outside the weight window must raise, not silently score 0.0."""
        train, actuals, preds = self._panel(100, 50)
        train_early = train.assign(d=train['d'] - 1800)
        actuals_early = actuals.assign(d=actuals['d'] - 1800)
        preds_early = preds.assign(d=preds['d'] - 1800)
        with pytest.raises(ValueError, match="weight window"):
            metrics.compute_wrmsse(actuals_early, preds_early, train_early)


class TestRandomWalkNaiveRMSSE:
    """Calibration: a naive forecast on a random walk must give RMSSE ~= 1.

    This is the check that catches a mis-scaled denominator. A one-step-ahead naive
    forecast on a driftless random walk has expected squared error equal to the
    increment variance, which is exactly what the RMSSE denominator estimates, so the
    ratio must sit at 1. A single series is far too noisy to assert that tightly, so
    take the median across many independent walks.
    """

    def test_one_step_naive_on_random_walk_is_near_one(self):
        rng = np.random.default_rng(seed=20260818)
        n_series, steps = 300, 500

        rmsses = []
        for k in range(n_series):
            # Offset to a high positive level. compute_naive_baseline_errors trims each
            # series to its first `sales > 0` day — a rule meant for non-negative count
            # data — so a walk that dips below zero gets truncated and can end up with
            # too few points, yielding a zero denominator and an infinite RMSSE.
            walk = 500.0 + np.cumsum(rng.standard_normal(steps + 1))
            train = pd.DataFrame({
                'id': [f'RW{k}'] * steps,
                'd': np.arange(1, steps + 1),
                'sales': walk[:steps],
            })
            # one-step horizon: forecast the next point as the last observed one
            actuals = pd.DataFrame({'id': [f'RW{k}'], 'd': [steps + 1], 'sales': [walk[steps]]})
            preds = pd.DataFrame({'id': [f'RW{k}'], 'd': [steps + 1], 'pred': [walk[steps - 1]]})
            rmsses.append(metrics.compute_rmsse(actuals, preds, train)[f'RW{k}'])

        median_rmsse = float(np.median(rmsses))
        assert np.all(np.isfinite(rmsses)), "some RMSSE values were not finite"
        # |N(0,1)| has median ~0.674; RMSSE here is |one-step error| / sqrt(naive MSE),
        # so the median lands near 0.674 rather than 1.0. The MEAN SQUARE is the
        # quantity calibrated to 1.
        rms = float(np.sqrt(np.mean(np.square(rmsses))))
        assert 0.85 < rms < 1.15, (
            f"root-mean-square RMSSE over {n_series} random walks = {rms}; "
            "expected ~1.0. A value far from 1 means the naive-error denominator "
            "is mis-scaled."
        )
        assert 0.5 < median_rmsse < 0.9, (
            f"median RMSSE = {median_rmsse}, expected ~0.674 (median of |N(0,1)|)"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
