# =============================================================================
# Processors Module
# =============================================================================
# This module handles data transformation and enrichment:
#   - cleaner.py: Clean and standardize raw data
#   - validator.py: Validate data quality, flag issues
#   - trends.py: Calculate trends, YoY changes, moving averages
# =============================================================================

from .cleaner import clean_dataframe, standardize_dates, remove_duplicates
from .validator import validate_data, check_missing_values, detect_outliers
from .trends import calculate_yoy_change, calculate_mom_change, calculate_moving_average

__all__ = [
    # Cleaner
    "clean_dataframe",
    "standardize_dates",
    "remove_duplicates",
    # Validator
    "validate_data",
    "check_missing_values",
    "detect_outliers",
    # Trends
    "calculate_yoy_change",
    "calculate_mom_change",
    "calculate_moving_average",
]
