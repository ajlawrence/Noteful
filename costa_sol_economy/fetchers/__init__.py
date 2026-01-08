# =============================================================================
# Fetchers Module
# =============================================================================
# This module contains functions to fetch data from various sources:
#   - dataestur.py: Spanish tourism data (Dataestur API)
#   - eurostat.py: European statistics (Eurostat API)
#   - ine.py: Spanish national statistics (INE)
#   - news.py: Economic news from RSS feeds
#
# Each fetcher returns a pandas DataFrame in a standardized format.
# =============================================================================

from .dataestur import fetch_tourism_arrivals, fetch_tourism_nights, fetch_tourism_expenditure
from .eurostat import fetch_gdp_eurostat, fetch_employment_eurostat, fetch_unemployment_eurostat
from .ine import fetch_hotel_occupancy_ine, fetch_housing_prices_ine
from .news import fetch_economic_news

__all__ = [
    # Tourism
    "fetch_tourism_arrivals",
    "fetch_tourism_nights",
    "fetch_tourism_expenditure",
    # GDP
    "fetch_gdp_eurostat",
    # Employment
    "fetch_employment_eurostat",
    "fetch_unemployment_eurostat",
    # INE
    "fetch_hotel_occupancy_ine",
    "fetch_housing_prices_ine",
    # News
    "fetch_economic_news",
]
