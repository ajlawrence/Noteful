# =============================================================================
# Trend Calculator
# =============================================================================
# Functions to calculate economic trends and derived metrics:
#   - Year-over-Year (YoY) change
#   - Month-over-Month (MoM) change
#   - Quarter-over-Quarter (QoQ) change
#   - Moving averages
#   - Growth rates
#
# These calculations are essential for economic analysis and are used by:
#   - The AI layer for anomaly detection
#   - Visualization for trend charts
#   - Reports for highlighting changes
# =============================================================================

import logging
from typing import Optional, List, Union

import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# YEAR-OVER-YEAR (YoY) CHANGE
# =============================================================================

def calculate_yoy_change(
    df: pd.DataFrame,
    value_column: str = "value",
    date_column: str = "date",
    output_column: str = "yoy_change"
) -> pd.DataFrame:
    """
    Calculate Year-over-Year percentage change.

    YoY change compares the current value to the same period one year ago.
    This is the most common metric for economic data as it removes seasonality.

    Formula: ((current - previous_year) / previous_year) * 100

    Args:
        df: DataFrame with time series data
        value_column: Column containing values to compare
        date_column: Column containing dates
        output_column: Name for the new column with YoY values

    Returns:
        DataFrame with new YoY column added

    Example:
        df = calculate_yoy_change(tourism_df)
        # Now df has 'yoy_change' column with values like -5.2 (5.2% decrease)
    """
    if df.empty:
        logger.warning("Empty DataFrame, cannot calculate YoY change")
        return df

    if value_column not in df.columns:
        logger.warning(f"Value column '{value_column}' not found")
        return df

    df = df.copy()

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date
    df = df.sort_values(date_column)

    # Create a date column for 1 year ago
    df["_date_1y_ago"] = df[date_column] - pd.DateOffset(years=1)

    # Self-join to find matching dates from 1 year ago
    # We need to handle this carefully for monthly/quarterly data

    yoy_values = []

    for idx, row in df.iterrows():
        current_value = row[value_column]
        target_date = row["_date_1y_ago"]

        # Find the closest date within a reasonable range (±15 days)
        date_min = target_date - pd.Timedelta(days=15)
        date_max = target_date + pd.Timedelta(days=15)

        mask = (df[date_column] >= date_min) & (df[date_column] <= date_max)
        matching_rows = df[mask]

        if len(matching_rows) == 0:
            yoy_values.append(np.nan)
        else:
            # Get the closest date
            matching_rows = matching_rows.copy()
            matching_rows["_date_diff"] = abs(matching_rows[date_column] - target_date)
            closest_row = matching_rows.loc[matching_rows["_date_diff"].idxmin()]
            previous_value = closest_row[value_column]

            if pd.isna(previous_value) or previous_value == 0:
                yoy_values.append(np.nan)
            else:
                yoy_change = ((current_value - previous_value) / previous_value) * 100
                yoy_values.append(round(yoy_change, 2))

    df[output_column] = yoy_values

    # Clean up temporary column
    df = df.drop(columns=["_date_1y_ago"])

    logger.info(f"Calculated YoY change for {len(df)} rows")

    return df


# =============================================================================
# MONTH-OVER-MONTH (MoM) CHANGE
# =============================================================================

def calculate_mom_change(
    df: pd.DataFrame,
    value_column: str = "value",
    date_column: str = "date",
    output_column: str = "mom_change"
) -> pd.DataFrame:
    """
    Calculate Month-over-Month percentage change.

    MoM change compares the current value to the previous month.
    Useful for detecting short-term trends but includes seasonality.

    Formula: ((current - previous_month) / previous_month) * 100

    Args:
        df: DataFrame with time series data
        value_column: Column containing values
        date_column: Column containing dates
        output_column: Name for the new column

    Returns:
        DataFrame with new MoM column added

    Example:
        df = calculate_mom_change(tourism_df)
    """
    if df.empty:
        logger.warning("Empty DataFrame, cannot calculate MoM change")
        return df

    if value_column not in df.columns:
        logger.warning(f"Value column '{value_column}' not found")
        return df

    df = df.copy()

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date
    df = df.sort_values(date_column)

    # Calculate percentage change from previous row
    # This assumes data is sorted and evenly spaced (monthly)
    df[output_column] = df[value_column].pct_change() * 100
    df[output_column] = df[output_column].round(2)

    logger.info(f"Calculated MoM change for {len(df)} rows")

    return df


# =============================================================================
# QUARTER-OVER-QUARTER (QoQ) CHANGE
# =============================================================================

def calculate_qoq_change(
    df: pd.DataFrame,
    value_column: str = "value",
    date_column: str = "date",
    output_column: str = "qoq_change"
) -> pd.DataFrame:
    """
    Calculate Quarter-over-Quarter percentage change.

    QoQ change compares the current value to the previous quarter.
    Used for quarterly data like GDP.

    Args:
        df: DataFrame with time series data
        value_column: Column containing values
        date_column: Column containing dates
        output_column: Name for the new column

    Returns:
        DataFrame with new QoQ column added
    """
    if df.empty:
        return df

    if value_column not in df.columns:
        return df

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    # For quarterly data, we look back 1 period (3 months)
    # If data is monthly, we need to look back 3 periods
    df["_quarter"] = df[date_column].dt.to_period("Q")

    # Calculate change
    df[output_column] = df[value_column].pct_change() * 100
    df[output_column] = df[output_column].round(2)

    df = df.drop(columns=["_quarter"])

    return df


# =============================================================================
# MOVING AVERAGES
# =============================================================================

def calculate_moving_average(
    df: pd.DataFrame,
    value_column: str = "value",
    window: int = 3,
    output_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate moving average to smooth out short-term fluctuations.

    A moving average takes the average of the last N periods, which helps
    identify underlying trends by reducing noise in the data.

    Args:
        df: DataFrame with time series data
        value_column: Column containing values
        window: Number of periods to include in average
        output_column: Name for new column (default: "{value_column}_ma{window}")

    Returns:
        DataFrame with new moving average column

    Example:
        # Calculate 3-month moving average
        df = calculate_moving_average(tourism_df, window=3)
        # Now df has 'value_ma3' column
    """
    if df.empty:
        return df

    if value_column not in df.columns:
        return df

    df = df.copy()

    if output_column is None:
        output_column = f"{value_column}_ma{window}"

    # Calculate rolling mean
    df[output_column] = df[value_column].rolling(window=window, min_periods=1).mean()
    df[output_column] = df[output_column].round(2)

    logger.info(f"Calculated {window}-period moving average")

    return df


# =============================================================================
# GROWTH RATE
# =============================================================================

def calculate_growth_rate(
    df: pd.DataFrame,
    value_column: str = "value",
    periods: int = 1,
    output_column: str = "growth_rate",
    annualize: bool = False
) -> pd.DataFrame:
    """
    Calculate growth rate over a specified number of periods.

    Args:
        df: DataFrame with time series data
        value_column: Column containing values
        periods: Number of periods to look back
        output_column: Name for new column
        annualize: If True, convert to annual growth rate

    Returns:
        DataFrame with growth rate column

    Example:
        # Calculate 12-month growth rate (annualized)
        df = calculate_growth_rate(df, periods=12, annualize=True)
    """
    if df.empty:
        return df

    if value_column not in df.columns:
        return df

    df = df.copy()

    # Calculate percentage change over N periods
    df[output_column] = df[value_column].pct_change(periods=periods) * 100

    if annualize and periods != 12:
        # Annualize the growth rate
        # This converts monthly growth to equivalent annual growth
        df[output_column] = ((1 + df[output_column] / 100) ** (12 / periods) - 1) * 100

    df[output_column] = df[output_column].round(2)

    return df


# =============================================================================
# SEASONAL ADJUSTMENT (Simple method)
# =============================================================================

def calculate_seasonal_index(
    df: pd.DataFrame,
    value_column: str = "value",
    date_column: str = "date"
) -> pd.DataFrame:
    """
    Calculate simple seasonal indices.

    This uses a basic ratio-to-moving-average method to estimate
    seasonal patterns in the data.

    Args:
        df: DataFrame with time series data
        value_column: Column containing values
        date_column: Column containing dates

    Returns:
        DataFrame with seasonal_index column
    """
    if df.empty or len(df) < 13:  # Need at least 13 months
        logger.warning("Not enough data for seasonal adjustment")
        return df

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Calculate 12-month moving average
    df["_ma12"] = df[value_column].rolling(window=12, center=True, min_periods=1).mean()

    # Calculate ratio to moving average
    df["_ratio"] = df[value_column] / df["_ma12"]

    # Get month from date
    df["_month"] = df[date_column].dt.month

    # Calculate average ratio for each month
    monthly_indices = df.groupby("_month")["_ratio"].mean()

    # Normalize so average = 1
    monthly_indices = monthly_indices / monthly_indices.mean()

    # Map back to DataFrame
    df["seasonal_index"] = df["_month"].map(monthly_indices)

    # Calculate seasonally adjusted value
    df["value_sa"] = df[value_column] / df["seasonal_index"]

    # Clean up
    df = df.drop(columns=["_ma12", "_ratio", "_month"])

    return df


# =============================================================================
# TREND DETECTION
# =============================================================================

def detect_trend_direction(
    df: pd.DataFrame,
    value_column: str = "value",
    window: int = 6
) -> str:
    """
    Detect the overall trend direction in recent data.

    Uses linear regression on the last N periods to determine
    if the trend is up, down, or flat.

    Args:
        df: DataFrame with time series data
        value_column: Column containing values
        window: Number of recent periods to analyze

    Returns:
        One of: "upward", "downward", "flat", "insufficient_data"
    """
    if df.empty or len(df) < window:
        return "insufficient_data"

    if value_column not in df.columns:
        return "insufficient_data"

    # Get last N values
    recent_values = df[value_column].tail(window).dropna()

    if len(recent_values) < 3:
        return "insufficient_data"

    # Simple linear regression
    x = np.arange(len(recent_values))
    y = recent_values.values

    # Calculate slope
    slope = np.polyfit(x, y, 1)[0]

    # Normalize slope by average value
    avg_value = np.mean(y)
    if avg_value == 0:
        return "flat"

    normalized_slope = (slope / avg_value) * 100  # Percentage per period

    # Classify trend
    if normalized_slope > 1:  # More than 1% per period
        return "upward"
    elif normalized_slope < -1:  # Less than -1% per period
        return "downward"
    else:
        return "flat"


def summarize_trends(
    df: pd.DataFrame,
    value_column: str = "value",
    date_column: str = "date"
) -> dict:
    """
    Generate a summary of trends in the data.

    Args:
        df: DataFrame with time series data
        value_column: Column containing values
        date_column: Column containing dates

    Returns:
        Dictionary with trend summary
    """
    if df.empty:
        return {"error": "Empty DataFrame"}

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    # Calculate various metrics
    current_value = df[value_column].iloc[-1] if not df[value_column].isna().iloc[-1] else None

    # YoY change for most recent period
    df_with_yoy = calculate_yoy_change(df, value_column, date_column)
    latest_yoy = df_with_yoy["yoy_change"].iloc[-1] if "yoy_change" in df_with_yoy.columns else None

    # MoM change
    df_with_mom = calculate_mom_change(df, value_column, date_column)
    latest_mom = df_with_mom["mom_change"].iloc[-1] if "mom_change" in df_with_mom.columns else None

    # Trend direction
    trend = detect_trend_direction(df, value_column)

    # Historical high/low
    all_time_high = df[value_column].max()
    all_time_low = df[value_column].min()

    # Is current value at all-time high/low?
    at_high = current_value == all_time_high if current_value else False
    at_low = current_value == all_time_low if current_value else False

    return {
        "current_value": current_value,
        "yoy_change_percent": latest_yoy,
        "mom_change_percent": latest_mom,
        "trend_direction": trend,
        "all_time_high": all_time_high,
        "all_time_low": all_time_low,
        "at_all_time_high": at_high,
        "at_all_time_low": at_low,
        "data_points": len(df),
        "date_range": {
            "start": str(df[date_column].min())[:10],
            "end": str(df[date_column].max())[:10]
        }
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Trend Calculator...")

    # Create test data - 24 months of tourism data with seasonal pattern
    dates = pd.date_range("2023-01-01", periods=24, freq="M")
    base_values = [100, 95, 110, 140, 180, 200, 220, 210, 170, 130, 105, 100]
    values = base_values + [v * 1.05 for v in base_values]  # 5% growth year 2

    test_df = pd.DataFrame({
        "date": dates,
        "indicator": "tourist_arrivals",
        "value": values,
        "location": "Málaga"
    })

    print("\nTest Data (24 months):")
    print(test_df.head(12))

    # Calculate trends
    print("\n--- Calculating YoY Change ---")
    df_with_yoy = calculate_yoy_change(test_df)
    print(df_with_yoy[["date", "value", "yoy_change"]].tail(6))

    print("\n--- Calculating MoM Change ---")
    df_with_mom = calculate_mom_change(test_df)
    print(df_with_mom[["date", "value", "mom_change"]].tail(6))

    print("\n--- Calculating Moving Average ---")
    df_with_ma = calculate_moving_average(test_df, window=3)
    print(df_with_ma[["date", "value", "value_ma3"]].tail(6))

    print("\n--- Trend Summary ---")
    summary = summarize_trends(test_df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
