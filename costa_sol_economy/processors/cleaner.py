# =============================================================================
# Data Cleaner
# =============================================================================
# Functions to clean and standardize data from various sources.
#
# Raw data from APIs often has:
#   - Inconsistent date formats
#   - Missing values represented differently (NaN, "", "N/A", etc.)
#   - Duplicate records
#   - Inconsistent naming conventions
#
# This module standardizes everything for consistent storage and analysis.
# =============================================================================

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# MAIN CLEANING FUNCTION
# =============================================================================

def clean_dataframe(
    df: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "value",
    required_columns: Optional[List[str]] = None,
    fill_missing_values: bool = False,
    remove_negative: bool = False
) -> pd.DataFrame:
    """
    Clean a DataFrame by applying standard transformations.

    This is the main cleaning function that applies multiple operations:
    1. Standardize date format
    2. Convert values to numeric
    3. Remove duplicates
    4. Handle missing values
    5. Sort by date

    Args:
        df: Input DataFrame to clean
        date_column: Name of the date column
        value_column: Name of the value column
        required_columns: List of columns that must be present
        fill_missing_values: If True, forward-fill missing values
        remove_negative: If True, remove rows with negative values

    Returns:
        Cleaned DataFrame

    Example:
        raw_df = fetch_tourism_arrivals()
        clean_df = clean_dataframe(raw_df, date_column="date", value_column="value")
    """
    if df.empty:
        logger.warning("Received empty DataFrame, nothing to clean")
        return df

    # Make a copy to avoid modifying the original
    df = df.copy()

    logger.info(f"Cleaning DataFrame with {len(df)} rows")

    # Step 1: Standardize dates
    if date_column in df.columns:
        df = standardize_dates(df, date_column)

    # Step 2: Convert values to numeric
    if value_column in df.columns:
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    # Step 3: Remove duplicates
    df = remove_duplicates(df, date_column=date_column)

    # Step 4: Handle missing values
    if fill_missing_values and value_column in df.columns:
        df[value_column] = df[value_column].ffill()

    # Step 5: Remove negative values if requested
    if remove_negative and value_column in df.columns:
        initial_len = len(df)
        df = df[df[value_column] >= 0]
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with negative values")

    # Step 6: Sort by date
    if date_column in df.columns:
        df = df.sort_values(date_column)

    # Step 7: Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Cleaning complete. Resulting DataFrame has {len(df)} rows")

    return df


# =============================================================================
# DATE STANDARDIZATION
# =============================================================================

def standardize_dates(
    df: pd.DataFrame,
    date_column: str = "date",
    output_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Standardize date column to consistent datetime format.

    Handles various input formats:
    - "2024-01-15" (ISO format)
    - "15/01/2024" (European format)
    - "January 2024"
    - "2024Q1" (quarterly)
    - "2024M01" (monthly)

    Args:
        df: Input DataFrame
        date_column: Name of the date column
        output_format: Optional strftime format for output
                      If None, keeps as datetime

    Returns:
        DataFrame with standardized date column

    Example:
        df = standardize_dates(df, date_column="period")
    """
    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found in DataFrame")
        return df

    df = df.copy()

    # Try to parse dates
    original_dates = df[date_column].copy()

    # First, try pandas automatic parsing
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # For any that failed, try custom parsing
    failed_mask = df[date_column].isna() & original_dates.notna()

    if failed_mask.any():
        for idx in df[failed_mask].index:
            original_value = str(original_dates.loc[idx])
            parsed = _parse_custom_date(original_value)
            if parsed:
                df.loc[idx, date_column] = parsed

    # Log any remaining failures
    still_failed = df[date_column].isna() & original_dates.notna()
    if still_failed.any():
        logger.warning(
            f"{still_failed.sum()} dates could not be parsed. "
            f"Examples: {original_dates[still_failed].head(3).tolist()}"
        )

    # Convert to specified format if requested
    if output_format:
        df[date_column] = df[date_column].dt.strftime(output_format)

    return df


def _parse_custom_date(date_string: str) -> Optional[datetime]:
    """
    Parse non-standard date formats.

    Args:
        date_string: Date string to parse

    Returns:
        datetime object or None if parsing fails
    """
    date_string = str(date_string).strip()

    # Handle quarterly format: "2024Q1", "2024-Q1"
    if "Q" in date_string.upper():
        try:
            date_string = date_string.upper().replace("-", "")
            year = int(date_string[:4])
            quarter = int(date_string.replace("Q", "")[-1])
            month = (quarter - 1) * 3 + 1
            return datetime(year, month, 1)
        except (ValueError, IndexError):
            pass

    # Handle monthly format: "2024M01", "2024-M01"
    if "M" in date_string.upper():
        try:
            date_string = date_string.upper().replace("-", "")
            year = int(date_string[:4])
            month = int(date_string.split("M")[1][:2])
            return datetime(year, month, 1)
        except (ValueError, IndexError):
            pass

    # Handle Spanish month names
    spanish_months = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }

    lower_str = date_string.lower()
    for month_name, month_num in spanish_months.items():
        if month_name in lower_str:
            # Extract year (4 digits)
            import re
            year_match = re.search(r'\d{4}', date_string)
            if year_match:
                year = int(year_match.group())
                return datetime(year, month_num, 1)

    return None


# =============================================================================
# DUPLICATE REMOVAL
# =============================================================================

def remove_duplicates(
    df: pd.DataFrame,
    date_column: str = "date",
    indicator_column: str = "indicator",
    location_column: str = "location",
    keep: str = "last"
) -> pd.DataFrame:
    """
    Remove duplicate records from DataFrame.

    Duplicates are identified by matching date, indicator, and location.
    By default, keeps the most recent record (last occurrence).

    Args:
        df: Input DataFrame
        date_column: Name of date column
        indicator_column: Name of indicator column
        location_column: Name of location column
        keep: Which duplicate to keep ("first", "last", or False to drop all)

    Returns:
        DataFrame with duplicates removed

    Example:
        df = remove_duplicates(df)
        # Keeps only unique date/indicator/location combinations
    """
    initial_len = len(df)

    # Build list of columns to check for duplicates
    subset_cols = []
    for col in [date_column, indicator_column, location_column]:
        if col in df.columns:
            subset_cols.append(col)

    if not subset_cols:
        # No relevant columns, just remove exact duplicates
        df = df.drop_duplicates(keep=keep)
    else:
        df = df.drop_duplicates(subset=subset_cols, keep=keep)

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")

    return df


# =============================================================================
# VALUE CLEANING
# =============================================================================

def clean_numeric_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    replace_negative_with: Optional[float] = None,
    cap_outliers: bool = False,
    outlier_std: float = 3.0
) -> pd.DataFrame:
    """
    Clean numeric value columns.

    Args:
        df: Input DataFrame
        columns: List of columns to clean (default: all numeric columns)
        replace_negative_with: Replace negative values with this (None = keep)
        cap_outliers: If True, cap values beyond outlier_std standard deviations
        outlier_std: Number of standard deviations for outlier detection

    Returns:
        DataFrame with cleaned numeric values
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace negative values
        if replace_negative_with is not None:
            negative_mask = df[col] < 0
            if negative_mask.any():
                logger.info(f"Replacing {negative_mask.sum()} negative values in '{col}'")
                df.loc[negative_mask, col] = replace_negative_with

        # Cap outliers
        if cap_outliers:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - (outlier_std * std)
            upper_bound = mean + (outlier_std * std)

            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            if outliers.any():
                logger.info(f"Capping {outliers.sum()} outliers in '{col}'")
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound

    return df


# =============================================================================
# STRING CLEANING
# =============================================================================

def clean_string_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lowercase: bool = False,
    strip_whitespace: bool = True
) -> pd.DataFrame:
    """
    Clean string/text columns.

    Args:
        df: Input DataFrame
        columns: List of columns to clean (default: all object columns)
        lowercase: Convert to lowercase
        strip_whitespace: Remove leading/trailing whitespace

    Returns:
        DataFrame with cleaned string values
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        # Strip whitespace
        if strip_whitespace:
            df[col] = df[col].astype(str).str.strip()

        # Convert to lowercase
        if lowercase:
            df[col] = df[col].str.lower()

        # Replace empty strings with NaN
        df[col] = df[col].replace("", np.nan)
        df[col] = df[col].replace("nan", np.nan)

    return df


# =============================================================================
# LOCATION STANDARDIZATION
# =============================================================================

def standardize_locations(
    df: pd.DataFrame,
    location_column: str = "location"
) -> pd.DataFrame:
    """
    Standardize location names to consistent format.

    Maps various spellings to standard names:
    - "Málaga" / "Malaga" / "MÁLAGA" -> "Málaga"
    - "Andalucía" / "Andalucia" / "ANDALUCÍA" -> "Andalucía"
    - "Costa del Sol" / "costa del sol" -> "Costa del Sol"

    Args:
        df: Input DataFrame
        location_column: Name of location column

    Returns:
        DataFrame with standardized location names
    """
    if location_column not in df.columns:
        return df

    df = df.copy()

    # Mapping of variations to standard names
    location_map = {
        # Málaga variations
        "malaga": "Málaga",
        "málaga": "Málaga",
        "MALAGA": "Málaga",
        "MÁLAGA": "Málaga",
        "provincia de málaga": "Málaga",
        "provincia de malaga": "Málaga",
        # Andalucía variations
        "andalucia": "Andalucía",
        "andalucía": "Andalucía",
        "ANDALUCIA": "Andalucía",
        "ANDALUCÍA": "Andalucía",
        "comunidad autónoma de andalucía": "Andalucía",
        # Costa del Sol
        "costa del sol": "Costa del Sol",
        "COSTA DEL SOL": "Costa del Sol",
        # Spain
        "españa": "España",
        "spain": "España",
        "ESPAÑA": "España",
        "es": "España",
    }

    # Apply mapping
    df[location_column] = df[location_column].str.strip()
    df[location_column] = df[location_column].replace(location_map)

    return df


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Data Cleaner...")

    # Create test data
    test_df = pd.DataFrame({
        "date": ["2024-01-15", "2024Q1", "January 2024", "2024M02", "invalid"],
        "indicator": ["arrivals", "arrivals", "arrivals", "arrivals", "arrivals"],
        "value": [100, 200, 300, -50, "N/A"],
        "location": ["malaga", "MÁLAGA", "Malaga", "Málaga", "málaga"]
    })

    print("\nOriginal DataFrame:")
    print(test_df)

    # Clean it
    clean_df = clean_dataframe(test_df)
    clean_df = standardize_locations(clean_df)

    print("\nCleaned DataFrame:")
    print(clean_df)
