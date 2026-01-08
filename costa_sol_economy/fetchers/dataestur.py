# =============================================================================
# Dataestur API Fetcher
# =============================================================================
# Dataestur is Spain's official tourism statistics portal.
# It provides data on:
#   - Tourist arrivals (Frontur)
#   - Tourist expenditure (Egatur)
#   - Overnight stays
#
# API Documentation: https://www.dataestur.es
#
# NOTE: The Dataestur API structure may change. This module includes
# fallback methods to download data directly from their data explorer.
# =============================================================================

import logging
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base URL for Dataestur API
DATAESTUR_BASE_URL = "https://www.dataestur.es"

# Region codes we're interested in
ANDALUCIA_CODE = "01"  # Autonomous community code
MALAGA_PROVINCE = "29"  # Province code

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# Headers to mimic a browser (some APIs block non-browser requests)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CostaEconMonitor/1.0)",
    "Accept": "application/json",
}


# =============================================================================
# RETRY DECORATOR
# =============================================================================
# This decorator automatically retries failed API calls
# - Tries up to 3 times
# - Waits longer between each retry (exponential backoff)
# =============================================================================

def create_retry_decorator():
    """Create a retry decorator for API calls."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _make_api_request(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None
) -> Optional[Dict]:
    """
    Make a request to the Dataestur API.

    Args:
        endpoint: API endpoint (will be appended to base URL)
        params: Query parameters

    Returns:
        JSON response as dictionary, or None if request failed
    """
    url = f"{DATAESTUR_BASE_URL}{endpoint}"

    try:
        logger.info(f"Fetching from Dataestur: {url}")
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Dataestur API request failed: {e}")
        return None


def _parse_dataestur_response(
    data: Dict,
    indicator_name: str
) -> pd.DataFrame:
    """
    Parse Dataestur API response into a standardized DataFrame.

    The Dataestur API returns data in various formats depending on the endpoint.
    This function attempts to normalize the response.

    Args:
        data: Raw API response
        indicator_name: Name to assign to the indicator column

    Returns:
        DataFrame with columns: date, indicator, value, location
    """
    records = []

    # The API response structure varies - try common patterns
    try:
        # Pattern 1: data.series[].values[]
        if "series" in data:
            for series in data.get("series", []):
                location = series.get("territory", {}).get("name", "Andalucía")
                for point in series.get("values", []):
                    records.append({
                        "date": point.get("period", point.get("date")),
                        "indicator": indicator_name,
                        "value": point.get("value"),
                        "location": location
                    })

        # Pattern 2: data.data[] (flat array)
        elif "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                records.append({
                    "date": item.get("periodo", item.get("date")),
                    "indicator": indicator_name,
                    "value": item.get("valor", item.get("value")),
                    "location": item.get("territorio", "Andalucía")
                })

        # Pattern 3: Direct results array
        elif "results" in data:
            for item in data["results"]:
                records.append({
                    "date": item.get("date"),
                    "indicator": indicator_name,
                    "value": item.get("value"),
                    "location": item.get("location", "Andalucía")
                })

    except (KeyError, TypeError) as e:
        logger.warning(f"Could not parse Dataestur response: {e}")

    if not records:
        logger.warning("No records parsed from Dataestur response")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Clean up the date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    return df


# =============================================================================
# MAIN FETCHER FUNCTIONS
# =============================================================================

@create_retry_decorator()
def fetch_tourism_arrivals(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    region: str = "andalucia"
) -> pd.DataFrame:
    """
    Fetch tourist arrival data from Dataestur (Frontur survey).

    Frontur tracks international tourists entering Spain by region.

    Args:
        start_year: Start year for data (default: 5 years ago)
        end_year: End year for data (default: current year)
        region: Region to fetch ("andalucia" or "malaga")

    Returns:
        DataFrame with columns:
        - date: Period (monthly)
        - indicator: "tourist_arrivals"
        - value: Number of arrivals
        - location: Region name
        - unit: "persons"

    Example:
        df = fetch_tourism_arrivals(start_year=2020)
        print(df.head())
    """
    # Set default date range
    current_year = datetime.now().year
    start_year = start_year or (current_year - 5)
    end_year = end_year or current_year

    logger.info(f"Fetching tourist arrivals for {region} ({start_year}-{end_year})")

    # Try the Dataestur API first
    # Note: The exact endpoint may need adjustment based on current API structure
    params = {
        "territory": "01" if region == "andalucia" else "29",  # Andalucía or Málaga
        "startYear": start_year,
        "endYear": end_year,
        "frequency": "monthly"
    }

    # Attempt API fetch
    data = _make_api_request("/api/frontur/arrivals", params)

    if data:
        df = _parse_dataestur_response(data, "tourist_arrivals")
        if not df.empty:
            df["unit"] = "persons"
            df["source"] = "dataestur"
            return df

    # If API fails, try to generate sample structure for testing
    # In production, this would be replaced with actual CSV download or alternative API
    logger.warning("Dataestur API unavailable - returning empty DataFrame")
    logger.info("TIP: You can manually download data from https://www.dataestur.es/datos/")

    # Return empty DataFrame with correct structure
    return pd.DataFrame(columns=[
        "date", "indicator", "value", "location", "unit", "source"
    ])


@create_retry_decorator()
def fetch_tourism_nights(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    region: str = "andalucia"
) -> pd.DataFrame:
    """
    Fetch overnight stays data from Dataestur.

    Tracks the number of nights tourists spend in accommodations.

    Args:
        start_year: Start year for data
        end_year: End year for data
        region: Region to fetch

    Returns:
        DataFrame with columns:
        - date: Period (monthly)
        - indicator: "overnight_stays"
        - value: Number of nights
        - location: Region name
        - unit: "nights"
    """
    current_year = datetime.now().year
    start_year = start_year or (current_year - 5)
    end_year = end_year or current_year

    logger.info(f"Fetching overnight stays for {region} ({start_year}-{end_year})")

    params = {
        "territory": "01" if region == "andalucia" else "29",
        "startYear": start_year,
        "endYear": end_year,
        "frequency": "monthly"
    }

    data = _make_api_request("/api/egatur/nights", params)

    if data:
        df = _parse_dataestur_response(data, "overnight_stays")
        if not df.empty:
            df["unit"] = "nights"
            df["source"] = "dataestur"
            return df

    logger.warning("Dataestur nights API unavailable - returning empty DataFrame")
    return pd.DataFrame(columns=[
        "date", "indicator", "value", "location", "unit", "source"
    ])


@create_retry_decorator()
def fetch_tourism_expenditure(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    region: str = "andalucia"
) -> pd.DataFrame:
    """
    Fetch tourist expenditure data from Dataestur (Egatur survey).

    Egatur tracks how much money tourists spend in Spain.

    Args:
        start_year: Start year for data
        end_year: End year for data
        region: Region to fetch

    Returns:
        DataFrame with columns:
        - date: Period (monthly/quarterly)
        - indicator: "tourist_expenditure" or "expenditure_per_tourist"
        - value: Amount in EUR
        - location: Region name
        - unit: "EUR" or "EUR_per_person"
    """
    current_year = datetime.now().year
    start_year = start_year or (current_year - 5)
    end_year = end_year or current_year

    logger.info(f"Fetching tourist expenditure for {region} ({start_year}-{end_year})")

    params = {
        "territory": "01" if region == "andalucia" else "29",
        "startYear": start_year,
        "endYear": end_year,
    }

    data = _make_api_request("/api/egatur/expenditure", params)

    if data:
        df = _parse_dataestur_response(data, "tourist_expenditure")
        if not df.empty:
            df["unit"] = "EUR"
            df["source"] = "dataestur"
            return df

    logger.warning("Dataestur expenditure API unavailable - returning empty DataFrame")
    return pd.DataFrame(columns=[
        "date", "indicator", "value", "location", "unit", "source"
    ])


# =============================================================================
# FALLBACK: Manual CSV Import
# =============================================================================

def import_tourism_csv(
    file_path: str,
    indicator_name: str,
    date_column: str = "date",
    value_column: str = "value",
    location: str = "Málaga"
) -> pd.DataFrame:
    """
    Import tourism data from a manually downloaded CSV file.

    Use this as a fallback when APIs are unavailable.
    Download CSVs from: https://www.dataestur.es/datos/

    Args:
        file_path: Path to the CSV file
        indicator_name: Name for the indicator column
        date_column: Name of the date column in the CSV
        value_column: Name of the value column in the CSV
        location: Location name to assign

    Returns:
        DataFrame in standardized format

    Example:
        df = import_tourism_csv(
            "downloads/frontur_andalucia_2024.csv",
            indicator_name="tourist_arrivals",
            date_column="Periodo",
            value_column="Valor"
        )
    """
    try:
        # Read the CSV
        raw_df = pd.read_csv(file_path)

        # Rename columns to standard names
        df = pd.DataFrame({
            "date": pd.to_datetime(raw_df[date_column], errors="coerce"),
            "indicator": indicator_name,
            "value": pd.to_numeric(raw_df[value_column], errors="coerce"),
            "location": location,
            "source": "dataestur_csv"
        })

        # Remove rows with invalid dates or values
        df = df.dropna(subset=["date", "value"])

        logger.info(f"Imported {len(df)} records from {file_path}")
        return df

    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error importing CSV: {e}")
        return pd.DataFrame()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test of the fetchers
    logging.basicConfig(level=logging.INFO)

    print("Testing Dataestur fetchers...")
    print("\n1. Tourist Arrivals:")
    arrivals = fetch_tourism_arrivals(start_year=2023)
    print(f"   Records: {len(arrivals)}")
    if not arrivals.empty:
        print(arrivals.head())

    print("\n2. Overnight Stays:")
    nights = fetch_tourism_nights(start_year=2023)
    print(f"   Records: {len(nights)}")

    print("\n3. Tourist Expenditure:")
    expenditure = fetch_tourism_expenditure(start_year=2023)
    print(f"   Records: {len(expenditure)}")
