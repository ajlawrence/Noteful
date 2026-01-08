# =============================================================================
# Data Validator
# =============================================================================
# Functions to validate data quality and flag potential issues.
#
# Validation checks include:
#   - Missing values
#   - Outliers (statistical anomalies)
#   - Data type consistency
#   - Value range checks
#   - Temporal gaps
#
# This module is used BEFORE storing data to catch problems early.
# The AI layer can also use this to provide data quality insights.
# =============================================================================

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT CLASS
# =============================================================================

class ValidationResult:
    """
    Container for validation results.

    Attributes:
        is_valid: Overall validation passed (True/False)
        issues: List of issue descriptions
        warnings: List of warning messages
        stats: Dictionary of validation statistics
    """

    def __init__(self):
        self.is_valid = True
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}

    def add_issue(self, message: str):
        """Add a critical issue (fails validation)."""
        self.issues.append(message)
        self.is_valid = False
        logger.error(f"Validation issue: {message}")

    def add_warning(self, message: str):
        """Add a warning (doesn't fail validation but should be noted)."""
        self.warnings.append(message)
        logger.warning(f"Validation warning: {message}")

    def __repr__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"<ValidationResult: {status}, {len(self.issues)} issues, {len(self.warnings)} warnings>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "stats": self.stats
        }


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_data(
    df: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "value",
    max_missing_percent: float = 20.0,
    outlier_std_threshold: float = 3.0,
    check_temporal_gaps: bool = True,
    expected_frequency: Optional[str] = None  # "D", "W", "M", "Q", "Y"
) -> ValidationResult:
    """
    Perform comprehensive validation on a DataFrame.

    Runs multiple validation checks and returns a ValidationResult
    containing all issues, warnings, and statistics.

    Args:
        df: DataFrame to validate
        date_column: Name of date column
        value_column: Name of value column
        max_missing_percent: Maximum allowed percentage of missing values
        outlier_std_threshold: Number of std devs to consider as outlier
        check_temporal_gaps: Check for missing time periods
        expected_frequency: Expected data frequency (for gap detection)

    Returns:
        ValidationResult with all findings

    Example:
        result = validate_data(tourism_df)
        if not result.is_valid:
            print("Validation failed!")
            for issue in result.issues:
                print(f"  - {issue}")
    """
    result = ValidationResult()

    # Check if DataFrame is empty
    if df.empty:
        result.add_issue("DataFrame is empty")
        return result

    logger.info(f"Validating DataFrame with {len(df)} rows")

    # Store basic stats
    result.stats["row_count"] = len(df)
    result.stats["column_count"] = len(df.columns)

    # Check 1: Missing values
    missing_result = check_missing_values(df, value_column, max_missing_percent)
    if missing_result["has_issue"]:
        result.add_issue(missing_result["message"])
    elif missing_result["has_warning"]:
        result.add_warning(missing_result["message"])
    result.stats["missing_values"] = missing_result

    # Check 2: Outliers
    if value_column in df.columns:
        outlier_result = detect_outliers(df, value_column, outlier_std_threshold)
        if outlier_result["outlier_count"] > 0:
            result.add_warning(
                f"Found {outlier_result['outlier_count']} potential outliers in '{value_column}'"
            )
        result.stats["outliers"] = outlier_result

    # Check 3: Date validity
    if date_column in df.columns:
        date_result = check_date_validity(df, date_column)
        if not date_result["all_valid"]:
            result.add_issue(f"Found {date_result['invalid_count']} invalid dates")
        result.stats["dates"] = date_result

    # Check 4: Temporal gaps
    if check_temporal_gaps and date_column in df.columns:
        gap_result = check_temporal_gaps_func(df, date_column, expected_frequency)
        if gap_result["has_gaps"]:
            result.add_warning(f"Found {gap_result['gap_count']} gaps in time series")
        result.stats["temporal_gaps"] = gap_result

    # Check 5: Value range
    if value_column in df.columns:
        range_result = check_value_range(df, value_column)
        result.stats["value_range"] = range_result

        # Flag if all values are the same (suspicious)
        if range_result["min"] == range_result["max"] and len(df) > 1:
            result.add_warning(f"All values in '{value_column}' are identical: {range_result['min']}")

    # Check 6: Duplicate check
    dup_result = check_duplicates(df, date_column)
    if dup_result["duplicate_count"] > 0:
        result.add_warning(f"Found {dup_result['duplicate_count']} duplicate entries")
    result.stats["duplicates"] = dup_result

    logger.info(f"Validation complete: {result}")

    return result


# =============================================================================
# INDIVIDUAL VALIDATION CHECKS
# =============================================================================

def check_missing_values(
    df: pd.DataFrame,
    value_column: str = "value",
    max_percent: float = 20.0
) -> Dict[str, Any]:
    """
    Check for missing values in the value column.

    Args:
        df: DataFrame to check
        value_column: Column to check for missing values
        max_percent: Maximum acceptable percentage of missing values

    Returns:
        Dictionary with:
        - total_count: Total number of rows
        - missing_count: Number of missing values
        - missing_percent: Percentage of missing values
        - has_issue: True if exceeds max_percent
        - has_warning: True if > 0 but < max_percent
        - message: Description of finding
    """
    if value_column not in df.columns:
        return {
            "total_count": len(df),
            "missing_count": 0,
            "missing_percent": 0.0,
            "has_issue": False,
            "has_warning": False,
            "message": f"Column '{value_column}' not found"
        }

    total = len(df)
    missing = df[value_column].isna().sum()
    percent = (missing / total) * 100 if total > 0 else 0

    has_issue = percent > max_percent
    has_warning = percent > 0 and not has_issue

    if has_issue:
        message = f"Missing values ({percent:.1f}%) exceed threshold ({max_percent}%)"
    elif has_warning:
        message = f"Some missing values detected: {missing} ({percent:.1f}%)"
    else:
        message = "No missing values"

    return {
        "total_count": total,
        "missing_count": int(missing),
        "missing_percent": round(percent, 2),
        "has_issue": has_issue,
        "has_warning": has_warning,
        "message": message
    }


def detect_outliers(
    df: pd.DataFrame,
    value_column: str = "value",
    std_threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect statistical outliers in numeric column.

    Uses the standard deviation method:
    - Values more than std_threshold standard deviations from the mean
      are considered potential outliers.

    Args:
        df: DataFrame to check
        value_column: Column to check for outliers
        std_threshold: Number of standard deviations for outlier detection

    Returns:
        Dictionary with:
        - mean: Mean of values
        - std: Standard deviation
        - lower_bound: Lower threshold
        - upper_bound: Upper threshold
        - outlier_count: Number of outliers found
        - outlier_indices: List of row indices with outliers
        - outlier_values: List of outlier values
    """
    if value_column not in df.columns:
        return {
            "outlier_count": 0,
            "message": f"Column '{value_column}' not found"
        }

    values = pd.to_numeric(df[value_column], errors="coerce")
    values = values.dropna()

    if len(values) < 3:
        return {
            "outlier_count": 0,
            "message": "Not enough data points for outlier detection"
        }

    mean = values.mean()
    std = values.std()

    if std == 0:
        return {
            "mean": mean,
            "std": 0,
            "outlier_count": 0,
            "message": "Standard deviation is 0 (all values identical)"
        }

    lower_bound = mean - (std_threshold * std)
    upper_bound = mean + (std_threshold * std)

    # Find outliers
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outlier_indices = values[outlier_mask].index.tolist()
    outlier_values = values[outlier_mask].tolist()

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": len(outlier_indices),
        "outlier_indices": outlier_indices[:10],  # Limit to first 10
        "outlier_values": [round(v, 2) for v in outlier_values[:10]]
    }


def check_date_validity(
    df: pd.DataFrame,
    date_column: str = "date"
) -> Dict[str, Any]:
    """
    Check if all dates are valid.

    Args:
        df: DataFrame to check
        date_column: Column containing dates

    Returns:
        Dictionary with date validity information
    """
    if date_column not in df.columns:
        return {
            "all_valid": True,
            "message": f"Column '{date_column}' not found"
        }

    # Try to convert to datetime
    dates = pd.to_datetime(df[date_column], errors="coerce")

    invalid_count = dates.isna().sum()
    all_valid = invalid_count == 0

    # Get date range
    valid_dates = dates.dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
    else:
        min_date = None
        max_date = None

    # Check for future dates
    future_dates = (dates > datetime.now()).sum() if len(valid_dates) > 0 else 0

    return {
        "all_valid": all_valid,
        "invalid_count": int(invalid_count),
        "min_date": str(min_date)[:10] if min_date else None,
        "max_date": str(max_date)[:10] if max_date else None,
        "future_dates": int(future_dates)
    }


def check_temporal_gaps_func(
    df: pd.DataFrame,
    date_column: str = "date",
    expected_frequency: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check for gaps in time series data.

    Args:
        df: DataFrame to check
        date_column: Column containing dates
        expected_frequency: Expected frequency ("D", "W", "M", "Q", "Y")
                          If None, will try to infer

    Returns:
        Dictionary with gap information
    """
    if date_column not in df.columns:
        return {
            "has_gaps": False,
            "message": f"Column '{date_column}' not found"
        }

    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    dates = dates.sort_values().drop_duplicates()

    if len(dates) < 2:
        return {
            "has_gaps": False,
            "gap_count": 0,
            "message": "Not enough data points to check for gaps"
        }

    # Infer frequency if not provided
    if expected_frequency is None:
        # Calculate median gap
        gaps = dates.diff().dropna()
        median_gap = gaps.median()

        if median_gap <= timedelta(days=2):
            expected_frequency = "D"
        elif median_gap <= timedelta(days=10):
            expected_frequency = "W"
        elif median_gap <= timedelta(days=45):
            expected_frequency = "M"
        elif median_gap <= timedelta(days=100):
            expected_frequency = "Q"
        else:
            expected_frequency = "Y"

    # Define expected gap for each frequency
    expected_gaps = {
        "D": timedelta(days=1),
        "W": timedelta(days=7),
        "M": timedelta(days=31),
        "Q": timedelta(days=92),
        "Y": timedelta(days=366)
    }

    expected_gap = expected_gaps.get(expected_frequency, timedelta(days=31))
    tolerance = expected_gap * 1.5  # Allow some flexibility

    # Find gaps larger than expected
    actual_gaps = dates.diff().dropna()
    large_gaps = actual_gaps[actual_gaps > tolerance]

    missing_periods = []
    for i, gap in large_gaps.items():
        # Get the date before the gap
        gap_start = dates.loc[dates.index[dates.index.get_loc(i) - 1]]
        missing_periods.append({
            "after": str(gap_start)[:10],
            "gap_days": gap.days
        })

    return {
        "has_gaps": len(large_gaps) > 0,
        "gap_count": len(large_gaps),
        "inferred_frequency": expected_frequency,
        "missing_periods": missing_periods[:10]  # Limit to first 10
    }


def check_value_range(
    df: pd.DataFrame,
    value_column: str = "value"
) -> Dict[str, Any]:
    """
    Get value range statistics.

    Args:
        df: DataFrame to check
        value_column: Column to analyze

    Returns:
        Dictionary with value statistics
    """
    if value_column not in df.columns:
        return {"message": f"Column '{value_column}' not found"}

    values = pd.to_numeric(df[value_column], errors="coerce").dropna()

    if len(values) == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None
        }

    return {
        "min": round(float(values.min()), 2),
        "max": round(float(values.max()), 2),
        "mean": round(float(values.mean()), 2),
        "median": round(float(values.median()), 2),
        "std": round(float(values.std()), 2),
        "count": len(values)
    }


def check_duplicates(
    df: pd.DataFrame,
    date_column: str = "date",
    indicator_column: str = "indicator"
) -> Dict[str, Any]:
    """
    Check for duplicate entries.

    Args:
        df: DataFrame to check
        date_column: Date column for duplicate detection
        indicator_column: Indicator column for duplicate detection

    Returns:
        Dictionary with duplicate information
    """
    subset_cols = []
    for col in [date_column, indicator_column]:
        if col in df.columns:
            subset_cols.append(col)

    if not subset_cols:
        # Check for exact duplicates
        duplicates = df.duplicated()
    else:
        duplicates = df.duplicated(subset=subset_cols)

    duplicate_count = duplicates.sum()

    return {
        "duplicate_count": int(duplicate_count),
        "duplicate_indices": df[duplicates].index.tolist()[:10]
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Data Validator...")

    # Create test data with various issues
    test_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="M").tolist() + [None],
        "indicator": ["arrivals"] * 13,
        "value": [100, 110, 105, 120, 115, 130, 1000, 140, 135, 150, 145, None, 160],
        "location": ["Málaga"] * 13
    })

    print("\nTest DataFrame:")
    print(test_df)

    # Validate
    result = validate_data(test_df, max_missing_percent=10.0)

    print(f"\nValidation Result: {result}")
    print(f"\nIs Valid: {result.is_valid}")

    if result.issues:
        print("\nIssues:")
        for issue in result.issues:
            print(f"  - {issue}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print("\nStatistics:")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")
