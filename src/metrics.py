"""
Competition metrics for M5 Forecasting Accuracy.

This module computes WRMSSE (Weighted Root Mean Squared Scaled Error) and supporting
diagnostics. All metric functions take actuals and predictions as explicit arguments —
they do not read data internally, ensuring leak-safety and reusability.

Key invariant: weights use the final 28 training days only (d_1886–d_1913),
never the holdout (d_1914–d_1941).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


# Training period definition
TRAIN_END = 1913  # Last day of training period
HORIZON_LEN = 28  # Days ahead to forecast
WEIGHT_PERIOD_START = TRAIN_END - HORIZON_LEN + 1  # d_1886
WEIGHT_PERIOD_END = TRAIN_END  # d_1913


def compute_naive_baseline_errors(
    actuals: pd.DataFrame,
    training_end: int = TRAIN_END,
) -> pd.Series:
    """
    Compute mean squared one-step-ahead errors for naive (random walk) forecast.

    For each series, compute the mean squared difference (y_t - y_{t-1})^2 over
    the training period, starting from the first non-zero sale (not from day 1).

    Parameters
    ----------
    actuals : pd.DataFrame
        Long-format data with columns: 'id' (series identifier), 'd' (day), 'sales'.
        Must include only training days (d <= training_end). Rows with d > training_end
        are filtered out before processing.

    training_end : int
        Last training day (default 1913 for this competition). Only days d <= training_end
        are used for computing the naive baseline error.

    Returns
    -------
    pd.Series
        Index: 'id' (unique per series), values: mean squared naive error.
        Series with zero non-zero sales across the training period will have 0 error
        (no differences to compute), not NaN — these are degenerate but handled gracefully.
    """
    actuals = actuals.copy()

    # Filter to training period only
    actuals = actuals[actuals["d"] <= training_end]

    # Sort by id and d for within-group operations
    actuals = actuals.sort_values(["id", "d"])

    # Find first non-zero day for each series
    first_nonzero_days = (
        actuals[actuals["sales"] > 0]
        .groupby("id", observed=True)["d"]
        .min()
    )

    # Broadcast first non-zero day to all rows via map; NaN for series with no non-zeros
    actuals["first_nz_day"] = actuals["id"].map(first_nonzero_days)

    # Keep only rows at or after first non-zero day (comparison with NaN -> False, excludes all-zero series)
    trimmed = actuals[actuals["d"] >= actuals["first_nz_day"]].copy()

    # Compute within-group differences.
    # Cast to float64 FIRST: `sales` is int16 on the real panel, and a daily change above
    # 181 units squares past int16's 32,767 limit and silently wraps. That would understate
    # the naive baseline — the RMSSE denominator — and so overstate every RMSSE built on it.
    trimmed["diff"] = (
        trimmed.assign(sales=trimmed["sales"].astype("float64"))
        .groupby("id", observed=True)["sales"]
        .diff()
    )

    # Square the differences
    trimmed["diff_sq"] = trimmed["diff"] ** 2

    # Compute mean of squared differences per series.
    # groupby.mean() ignores NaN, so the first row (NaN from diff) is skipped.
    # A series with only 1 row after trimming will have all NaN in diff_sq, so mean() -> NaN.
    mse_by_id = trimmed.groupby("id", observed=True)["diff_sq"].mean()

    # Replace NaN with 0.0 (handles series with <= 1 obs after trimming)
    mse_by_id = mse_by_id.fillna(0.0)

    # Reindex to include all original series, filling missing ones with 0.0
    # (series with no non-zero sales won't appear in trimmed, so won't be in mse_by_id)
    all_series = actuals["id"].unique()
    mse_by_id = mse_by_id.reindex(all_series, fill_value=0.0)

    return pd.Series(mse_by_id, name="naive_mse")


def compute_rmsse(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    training_actuals: pd.DataFrame,
) -> pd.Series:
    """
    Compute per-series RMSSE.

    RMSSE = sqrt(MSE / naive_mse)

    where MSE is the mean squared error over the forecast horizon and naive_mse
    is the mean squared one-step naive error over the training period.

    Parameters
    ----------
    actuals : pd.DataFrame
        Holdout actuals, long format: 'id', 'd', 'sales'. Should span horizon days.

    predictions : pd.DataFrame
        Predictions, same format: 'id', 'd', and one forecast column.
        Forecast column will be auto-detected as the only numeric column in predictions
        that is not 'id' or 'd'.

    training_actuals : pd.DataFrame
        Training period actuals: 'id', 'd', 'sales'. Used to compute naive MSE.

    Returns
    -------
    pd.Series
        Index: 'id', values: RMSSE for each series.
    """
    # Compute naive MSE from training data
    naive_mse = compute_naive_baseline_errors(training_actuals)

    # Identify the forecast column in predictions (should be only numeric non-id/d column)
    pred_cols = [c for c in predictions.columns if c not in ["id", "d"]]
    if len(pred_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 forecast column in predictions, found {len(pred_cols)}: {pred_cols}"
        )
    pred_col = pred_cols[0]

    # Merge actuals and predictions, keeping only id, d, sales, and pred_col
    merged = actuals[["id", "d", "sales"]].merge(
        predictions[["id", "d", pred_col]], on=["id", "d"], how="inner"
    )

    # Compute MSE by series
    merged["se"] = (merged["sales"] - merged[pred_col]) ** 2
    mse_by_id = merged.groupby("id")["se"].mean()

    # Align naive MSE with MSE by series
    naive_mse_aligned = naive_mse.reindex(mse_by_id.index)

    # Compute RMSSE with special handling for edge cases:
    # - If both MSE and naive_mse are 0: RMSSE = 0 (perfect predictions on constant/zero series)
    # - If naive_mse is 0 but MSE > 0: RMSSE = inf (predictions wrong on zero series)
    # - Otherwise: RMSSE = sqrt(MSE / naive_mse)
    # Vectorized: a per-series Python loop here costs minutes at level 12 (30,490 series),
    # and compute_rmsse is called once per aggregation level.
    mse_vals = mse_by_id.astype("float64")
    naive_vals = naive_mse_aligned.astype("float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        rmsse = np.sqrt(mse_vals / naive_vals)

    # Degenerate case: no variance in training (naive_mse == 0).
    # A NaN naive_mse (series absent from training) stays NaN, as before.
    degenerate = naive_vals == 0
    rmsse[degenerate & (mse_vals == 0)] = 0.0   # perfect forecast on a constant series
    rmsse[degenerate & (mse_vals > 0)] = np.inf  # wrong forecast on a constant series

    return rmsse


def compute_mae(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.Series:
    """
    Compute per-series Mean Absolute Error.

    Parameters
    ----------
    actuals : pd.DataFrame
        Holdout actuals: 'id', 'd', 'sales'.

    predictions : pd.DataFrame
        Predictions: 'id', 'd', and a forecast column.

    Returns
    -------
    pd.Series
        Index: 'id', values: MAE for each series.
    """
    # Identify the forecast column
    pred_cols = [c for c in predictions.columns if c not in ["id", "d"]]
    if len(pred_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 forecast column in predictions, found {len(pred_cols)}: {pred_cols}"
        )
    pred_col = pred_cols[0]

    merged = actuals[["id", "d", "sales"]].merge(
        predictions[["id", "d", pred_col]], on=["id", "d"], how="inner"
    )

    merged["ae"] = np.abs(merged["sales"] - merged[pred_col])
    mae = merged.groupby("id")["ae"].mean()

    return mae


def compute_mase(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    training_actuals: pd.DataFrame,
) -> pd.Series:
    """
    Compute per-series Mean Absolute Scaled Error.

    MASE = MAE / mean_absolute_naive_error

    where mean_absolute_naive_error is the MAE of a naive (random walk) forecast
    over the training period.

    Parameters
    ----------
    actuals : pd.DataFrame
        Holdout actuals: 'id', 'd', 'sales'.

    predictions : pd.DataFrame
        Predictions: 'id', 'd', and a forecast column.

    training_actuals : pd.DataFrame
        Training data: 'id', 'd', 'sales'.

    Returns
    -------
    pd.Series
        Index: 'id', values: MASE for each series.
    """
    # Compute MAE of naive forecast on training data
    training_copy = training_actuals.copy().sort_values(["id", "d"])
    training_copy["naive_forecast"] = training_copy.groupby("id")["sales"].shift(1)
    training_copy["ae"] = np.abs(training_copy["sales"] - training_copy["naive_forecast"])
    naive_mae = training_copy.groupby("id")["ae"].mean()

    # Compute MAE on holdout
    mae = compute_mae(actuals, predictions)

    # MASE = MAE / naive_mae
    # Check that all series in mae have a matching naive_mae
    naive_mae_aligned = naive_mae.reindex(mae.index)
    if naive_mae_aligned.isna().any():
        missing_series = mae.index[naive_mae_aligned.isna()].tolist()
        raise ValueError(
            f"Series in predictions have no matching naive_mae: {missing_series}"
        )

    mase = mae / naive_mae_aligned

    return mase


def compute_bias_pct(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
) -> float:
    """
    Compute total bias as a percentage of total actual sales.

    bias_pct = (sum(predictions) - sum(actuals)) / sum(actuals) * 100

    Parameters
    ----------
    actuals : pd.DataFrame
        Holdout actuals: 'id', 'd', 'sales'.

    predictions : pd.DataFrame
        Predictions: 'id', 'd', and a forecast column.

    Returns
    -------
    float
        Bias as a percentage.
    """
    # Identify the forecast column
    pred_cols = [c for c in predictions.columns if c not in ["id", "d"]]
    if len(pred_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 forecast column in predictions, found {len(pred_cols)}: {pred_cols}"
        )
    pred_col = pred_cols[0]

    merged = actuals[["id", "d", "sales"]].merge(
        predictions[["id", "d", pred_col]], on=["id", "d"], how="inner"
    )

    total_actual = merged["sales"].sum()
    total_pred = merged[pred_col].sum()

    if total_actual == 0:
        return 0.0

    return (total_pred - total_actual) / total_actual * 100


def compute_weights(
    sales_data: pd.DataFrame,
    prices_data: pd.DataFrame = None,
    weight_start: int = WEIGHT_PERIOD_START,
    weight_end: int = WEIGHT_PERIOD_END,
) -> pd.Series:
    """
    Compute series weights based on dollar sales share.

    Weights are each series' share of dollar sales (units × price) over the final
    28 days of the training period, normalized to sum to 1.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data: 'id', 'd', 'sales'. Should include only training data.
        If 'sell_price' column is present, will use it; otherwise defaults to price=1.0.

    prices_data : pd.DataFrame, optional
        Price data (not currently used; prices expected to be in sales_data already).

    weight_start : int
        First day of weight period (default 1886).

    weight_end : int
        Last day of weight period (default 1913).

    Returns
    -------
    pd.Series
        Index: 'id', values: normalized weights (sum to 1).
    """
    sales_copy = sales_data.copy()

    # Filter to weight period
    sales_copy = sales_copy[(sales_copy["d"] >= weight_start) & (sales_copy["d"] <= weight_end)]

    # Compute dollar sales (units × price)
    if "sell_price" in sales_copy.columns:
        sales_copy["dollar_sales"] = sales_copy["sales"] * sales_copy["sell_price"]
    else:
        sales_copy["dollar_sales"] = sales_copy["sales"]  # Default to units if no price

    # Sum by series
    dollar_by_id = sales_copy.groupby("id")["dollar_sales"].sum()

    # Normalize to sum to 1
    weights = dollar_by_id / dollar_by_id.sum()

    return weights


def compute_wrmsse(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    training_actuals: pd.DataFrame,
    level_grouping: List[List[str]] = None,
    weight_start: int = WEIGHT_PERIOD_START,
    weight_end: int = WEIGHT_PERIOD_END,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute WRMSSE (Weighted RMSSE) across all 12 aggregation levels.

    WRMSSE is the unweighted mean of the 12 per-level weighted RMSSE scores.

    Parameters
    ----------
    actuals : pd.DataFrame
        Holdout actuals: 'id', 'd', 'sales', plus grouping columns (cat_id, store_id, etc).

    predictions : pd.DataFrame
        Predictions: 'id', 'd', and a forecast column.

    training_actuals : pd.DataFrame
        Training data: 'id', 'd', 'sales', 'sell_price', plus grouping columns.

    level_grouping : list of list of str, optional
        List of grouping column combinations for each level.
        If None, uses the default 12 M5 levels.

    weight_start, weight_end : int
        Day range used to compute dollar-sales weights, defaulting to the final 28
        training days (d_1886–d_1913). Exposed mainly so tests can use toy day indices:
        with the competition defaults, any panel whose days fall outside 1886–1913
        yields no weight rows at all. Must stay inside the training period (invariant 1).

    Returns
    -------
    wrmsse : float
        Overall WRMSSE (mean of the 12 per-level scores).

    level_scores : dict
        Per-level WRMSSE scores keyed by level name.
    """
    if level_grouping is None:
        level_grouping = [
            [],  # Level 1: Total
            ["state_id"],  # Level 2: State
            ["store_id"],  # Level 3: Store
            ["cat_id"],  # Level 4: Category
            ["dept_id"],  # Level 5: Department
            ["state_id", "cat_id"],  # Level 6: State × Category
            ["state_id", "dept_id"],  # Level 7: State × Department
            ["store_id", "cat_id"],  # Level 8: Store × Category
            ["store_id", "dept_id"],  # Level 9: Store × Department
            ["item_id"],  # Level 10: Item
            ["item_id", "state_id"],  # Level 11: Item × State
            ["item_id", "store_id"],  # Level 12: Item × Store
        ]

    level_names = [
        "Total",
        "State",
        "Store",
        "Category",
        "Department",
        "State×Category",
        "State×Department",
        "Store×Category",
        "Store×Department",
        "Item",
        "Item×State",
        "Item×Store",
    ]

    # Identify forecast column
    pred_cols = [c for c in predictions.columns if c not in ["id", "d"]]
    if len(pred_cols) != 1:
        raise ValueError(f"Expected 1 forecast column, found {len(pred_cols)}: {pred_cols}")
    pred_col = pred_cols[0]

    level_scores = {}
    all_rmsse_scores = []

    for grouping_cols, level_name in zip(level_grouping, level_names):
        # Aggregate actuals to this level
        if grouping_cols:
            agg_act = (
                actuals.groupby(grouping_cols + ["d"])
                .agg({"sales": "sum"})
                .reset_index()
            )
            # Create level id from grouping columns using vectorized string concatenation.
            # Convert each column to string, then chain them with "_" separator.
            cols_as_str = [agg_act[col].astype(str) for col in grouping_cols]
            agg_act["level_id"] = cols_as_str[0]
            for col_str in cols_as_str[1:]:
                agg_act["level_id"] = agg_act["level_id"] + "_" + col_str
            agg_act = agg_act[["level_id", "d", "sales"]]
            agg_act = agg_act.rename(columns={"level_id": "id"})
        else:
            # Total level
            agg_act = actuals.groupby("d").agg({"sales": "sum"}).reset_index()
            agg_act["id"] = "TOTAL"
            agg_act = agg_act[["id", "d", "sales"]]

        # Aggregate predictions to this level
        if grouping_cols:
            # Note: drop_duplicates is required because actuals is long-format with one row per (id, d).
            # Without it, this many-to-many join fans out each prediction row by horizon_len,
            # inflating the aggregated sum by that factor and corrupting non-zero forecasts.
            # The id -> grouping columns mapping is exactly 1:1 in this dataset.
            agg_pred_data = predictions[["id", "d", pred_col]].merge(
                actuals[["id"] + grouping_cols].drop_duplicates(), on="id", how="left"
            )
            agg_pred = (
                agg_pred_data.groupby(grouping_cols + ["d"])
                .agg({pred_col: "sum"})
                .reset_index()
            )
            # Create level id from grouping columns using vectorized string concatenation.
            cols_as_str = [agg_pred[col].astype(str) for col in grouping_cols]
            agg_pred["level_id"] = cols_as_str[0]
            for col_str in cols_as_str[1:]:
                agg_pred["level_id"] = agg_pred["level_id"] + "_" + col_str
            agg_pred = agg_pred[["level_id", "d", pred_col]]
            agg_pred = agg_pred.rename(columns={"level_id": "id"})
        else:
            # Total level
            agg_pred = (
                predictions[[pred_col, "d"]].groupby("d").agg({pred_col: "sum"}).reset_index()
            )
            agg_pred["id"] = "TOTAL"
            agg_pred = agg_pred[["id", "d", pred_col]]

        # Aggregate training data the same way
        if grouping_cols:
            agg_train = (
                training_actuals.groupby(grouping_cols + ["d"])
                .agg({"sales": "sum"})
                .reset_index()
            )
            # Create level id from grouping columns using vectorized string concatenation.
            cols_as_str = [agg_train[col].astype(str) for col in grouping_cols]
            agg_train["level_id"] = cols_as_str[0]
            for col_str in cols_as_str[1:]:
                agg_train["level_id"] = agg_train["level_id"] + "_" + col_str
            agg_train = agg_train[["level_id", "d", "sales"]]
            agg_train = agg_train.rename(columns={"level_id": "id"})
        else:
            agg_train = training_actuals.groupby("d").agg({"sales": "sum"}).reset_index()
            agg_train["id"] = "TOTAL"
            agg_train = agg_train[["id", "d", "sales"]]

        # Compute RMSSE for aggregated series
        rmsse_agg = compute_rmsse(agg_act, agg_pred, agg_train)

        # Compute weights for this level from training data.
        # Fail loudly if the weight window selects nothing. Silently returning zero weights
        # makes every level score collapse to 0.0 regardless of the forecast, which looks
        # like a perfect result — the most dangerous failure this module can have.
        in_window = training_actuals[
            (training_actuals["d"] >= weight_start) & (training_actuals["d"] <= weight_end)
        ]
        if in_window.empty:
            raise ValueError(
                f"No training rows in the weight window d={weight_start}..{weight_end} "
                f"(training data spans d={training_actuals['d'].min()}..{training_actuals['d'].max()}). "
                "Weights would be undefined and every level score would collapse to 0.0. "
                "Pass weight_start/weight_end matching your data."
            )

        if grouping_cols:
            weight_data = in_window.copy()
            weight_data["dollar_sales"] = (
                weight_data["sales"] * weight_data["sell_price"]
            )
            # Create level id from grouping columns using vectorized string concatenation.
            cols_as_str = [weight_data[col].astype(str) for col in grouping_cols]
            weight_data["level_id"] = cols_as_str[0]
            for col_str in cols_as_str[1:]:
                weight_data["level_id"] = weight_data["level_id"] + "_" + col_str
            weights_level = (
                weight_data.groupby("level_id")["dollar_sales"].sum()
            )
            weights_level.index.name = None
        else:
            # Total level
            weight_data = in_window.copy()
            weight_data["dollar_sales"] = (
                weight_data["sales"] * weight_data["sell_price"]
            )
            weights_level = pd.Series({"TOTAL": weight_data["dollar_sales"].sum()})

        # Normalize weights to sum to 1
        if weights_level.empty:
            raise ValueError(f"Level '{level_name}' produced no weight groups.")
        if weights_level.sum() <= 0:
            raise ValueError(
                f"Level '{level_name}' has zero total dollar sales in the weight window "
                f"d={weight_start}..{weight_end}; weights are undefined."
            )
        weights_level = weights_level / weights_level.sum()

        # Compute weighted RMSSE for this level.
        # Check that all level groups in weights have a matching RMSSE value.
        rmsse_agg_aligned = rmsse_agg.reindex(weights_level.index)
        if rmsse_agg_aligned.isna().any():
            missing_groups = weights_level.index[rmsse_agg_aligned.isna()].tolist()
            missing_count = rmsse_agg_aligned.isna().sum()
            raise ValueError(
                f"Level '{level_name}': {missing_count} group(s) have weights but no RMSSE: {missing_groups}"
            )
        wrmsse_level = (rmsse_agg_aligned * weights_level).sum()

        level_scores[level_name] = wrmsse_level
        all_rmsse_scores.append(wrmsse_level)

    # Overall WRMSSE = unweighted mean of the 12 per-level scores
    wrmsse = np.mean(all_rmsse_scores)

    return wrmsse, level_scores


def compute_wmae(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    training_actuals: pd.DataFrame,
) -> float:
    """
    Compute weighted MAE (WMAE).

    WMAE = sum(weights * MAE for each series).

    Parameters
    ----------
    actuals : pd.DataFrame
        Holdout actuals: 'id', 'd', 'sales', 'sell_price'.

    predictions : pd.DataFrame
        Predictions: 'id', 'd', and a forecast column.

    training_actuals : pd.DataFrame
        Training data: 'id', 'd', 'sales', 'sell_price'.

    Returns
    -------
    float
        Weighted MAE.
    """
    mae = compute_mae(actuals, predictions)
    weights = compute_weights(training_actuals)

    # Align indices
    mae_aligned = mae.reindex(weights.index, fill_value=0)
    weights_aligned = weights.reindex(mae.index, fill_value=0)
    weights_aligned = weights_aligned / weights_aligned.sum()

    wmae = (mae_aligned * weights_aligned).sum()

    return wmae
