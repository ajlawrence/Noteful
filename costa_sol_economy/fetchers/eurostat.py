# =============================================================================
# Eurostat API Fetcher
# =============================================================================
# Eurostat is the statistical office of the European Union.
# It provides harmonized data across all EU countries.
#
# API Documentation: https://wikis.ec.europa.eu/display/EUROSTATHELP/API
#
# Key datasets we use:
#   - nama_10r_3gdp: Regional GDP at NUTS3 level
#   - nama_10r_3gva: Regional GVA by sector
#   - lfst_r_lfe2emprt: Regional employment rates
#   - lfst_r_lfu3rt: Regional unemployment rates
#
# Region codes:
#   - ES61: Andalucía (NUTS2)
#   - ES617: Málaga (NUTS3)
# =============================================================================

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Eurostat REST API base URL
EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# Our target regions
NUTS_CODES = {
    "andalucia": "ES61",  # NUTS2 - Autonomous Community
    "malaga": "ES617",    # NUTS3 - Province
    "spain": "ES"         # Country
}

# Request settings
REQUEST_TIMEOUT = 60  # Eurostat can be slow
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "CostaEconMonitor/1.0"
}


# =============================================================================
# RETRY DECORATOR
# =============================================================================

retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_eurostat_url(dataset: str, filters: Dict[str, Any]) -> str:
    """
    Build a Eurostat API URL with filters.

    The Eurostat API uses a specific URL structure:
    BASE/DATASET?param1=value1&param2=value2

    Args:
        dataset: Dataset code (e.g., "nama_10r_3gdp")
        filters: Dictionary of filter parameters

    Returns:
        Complete API URL
    """
    url = f"{EUROSTAT_API_BASE}/{dataset}"

    # Build query string from filters
    params = []
    for key, value in filters.items():
        if isinstance(value, list):
            # Multiple values: geo=ES61&geo=ES617
            for v in value:
                params.append(f"{key}={v}")
        else:
            params.append(f"{key}={value}")

    if params:
        url += "?" + "&".join(params)

    return url


@retry_decorator
def _fetch_eurostat_data(dataset: str, filters: Dict[str, Any]) -> Optional[Dict]:
    """
    Fetch data from the Eurostat API.

    Args:
        dataset: Dataset code
        filters: Query filters

    Returns:
        JSON response as dictionary, or None if failed
    """
    url = _build_eurostat_url(dataset, filters)

    try:
        logger.info(f"Fetching from Eurostat: {dataset}")
        logger.debug(f"URL: {url}")

        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Eurostat API request failed: {e}")
        return None


def _parse_eurostat_json(
    data: Dict,
    indicator_name: str,
    value_label: str = "value"
) -> pd.DataFrame:
    """
    Parse Eurostat JSON-stat format into a DataFrame.

    Eurostat uses the JSON-stat format which is compact but complex.
    The structure is:
    {
        "dimension": {
            "geo": {"category": {"index": {"ES617": 0}, "label": {"ES617": "Málaga"}}},
            "time": {"category": {"index": {"2020": 0, "2021": 1}, ...}},
            ...
        },
        "value": [123.4, 125.6, ...]  // Flat array of values
    }

    Args:
        data: Raw Eurostat JSON response
        indicator_name: Name for the indicator column
        value_label: Label for the value dimension

    Returns:
        DataFrame with columns: date, indicator, value, location
    """
    records = []

    try:
        # Extract dimensions
        dimensions = data.get("dimension", {})
        values = data.get("value", {})

        # Get time periods
        time_dim = dimensions.get("time", {}).get("category", {})
        time_index = time_dim.get("index", {})
        time_labels = time_dim.get("label", {})

        # Get geographic areas
        geo_dim = dimensions.get("geo", {}).get("category", {})
        geo_index = geo_dim.get("index", {})
        geo_labels = geo_dim.get("label", {})

        # Get the dimension sizes for index calculation
        dim_sizes = data.get("size", [])
        dim_ids = data.get("id", [])

        # Find positions of geo and time in dimensions
        geo_pos = dim_ids.index("geo") if "geo" in dim_ids else None
        time_pos = dim_ids.index("time") if "time" in dim_ids else None

        if geo_pos is None or time_pos is None:
            logger.warning("Could not find geo or time dimensions")
            return pd.DataFrame()

        # Iterate through all combinations
        for geo_code, geo_idx in geo_index.items():
            geo_label = geo_labels.get(geo_code, geo_code)

            for time_code, time_idx in time_index.items():
                # Calculate the flat index in the values array
                # This depends on the dimension order
                flat_idx = str(_calculate_flat_index(
                    dim_sizes, dim_ids, geo_pos, time_pos, geo_idx, time_idx
                ))

                # Get the value (might be None if no data)
                value = values.get(flat_idx)

                if value is not None:
                    records.append({
                        "date": _parse_eurostat_time(time_code),
                        "indicator": indicator_name,
                        "value": float(value),
                        "location": geo_label,
                        "location_code": geo_code
                    })

    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Error parsing Eurostat response: {e}")

    if not records:
        logger.warning("No records parsed from Eurostat response")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["source"] = "eurostat"

    return df


def _calculate_flat_index(
    dim_sizes: List[int],
    dim_ids: List[str],
    geo_pos: int,
    time_pos: int,
    geo_idx: int,
    time_idx: int
) -> int:
    """
    Calculate the flat array index from dimension indices.

    Eurostat stores values in a flat array, so we need to calculate
    the correct index based on the dimension structure.
    """
    # Simplified calculation - assumes geo and time are the main dimensions
    # In practice, you might need to handle more dimensions
    if len(dim_sizes) == 2:
        # Simple 2D case
        if dim_ids[0] == "geo":
            return geo_idx * dim_sizes[1] + time_idx
        else:
            return time_idx * dim_sizes[0] + geo_idx
    else:
        # More complex case - use a simpler heuristic
        # This may need adjustment for specific datasets
        total_idx = 0
        multiplier = 1
        for i in range(len(dim_sizes) - 1, -1, -1):
            if dim_ids[i] == "geo":
                total_idx += geo_idx * multiplier
            elif dim_ids[i] == "time":
                total_idx += time_idx * multiplier
            # Other dimensions assumed to be 0
            multiplier *= dim_sizes[i]
        return total_idx


def _parse_eurostat_time(time_code: str) -> datetime:
    """
    Parse Eurostat time period codes into datetime.

    Examples:
        "2023" -> 2023-01-01
        "2023Q1" -> 2023-01-01
        "2023M06" -> 2023-06-01
    """
    try:
        # Annual: "2023"
        if len(time_code) == 4 and time_code.isdigit():
            return datetime(int(time_code), 1, 1)

        # Quarterly: "2023Q1", "2023-Q1"
        if "Q" in time_code:
            year = int(time_code[:4])
            quarter = int(time_code[-1])
            month = (quarter - 1) * 3 + 1
            return datetime(year, month, 1)

        # Monthly: "2023M06", "2023-06"
        if "M" in time_code:
            year = int(time_code[:4])
            month = int(time_code.split("M")[1])
            return datetime(year, month, 1)

        if "-" in time_code and len(time_code) == 7:
            return datetime.strptime(time_code, "%Y-%m")

        # Default: try to parse as-is
        return pd.to_datetime(time_code)

    except (ValueError, TypeError):
        logger.warning(f"Could not parse time code: {time_code}")
        return pd.NaT


# =============================================================================
# MAIN FETCHER FUNCTIONS
# =============================================================================

def fetch_gdp_eurostat(
    regions: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch regional GDP data from Eurostat.

    Uses dataset: nama_10r_3gdp (Regional gross domestic product)

    Args:
        regions: List of region codes (default: ["ES617"] for Málaga)
        start_year: Start year (default: 10 years ago)
        end_year: End year (default: current year)

    Returns:
        DataFrame with columns:
        - date: Year (annual data)
        - indicator: "gdp_total"
        - value: GDP in millions of EUR
        - location: Region name
        - location_code: NUTS code
        - unit: "MIO_EUR"

    Example:
        df = fetch_gdp_eurostat(regions=["ES617", "ES61"])
        print(df.head())
    """
    # Set defaults
    regions = regions or ["ES617"]
    current_year = datetime.now().year
    start_year = start_year or (current_year - 10)
    end_year = end_year or current_year

    logger.info(f"Fetching GDP data for regions: {regions}")

    # Build filters
    filters = {
        "geo": regions,
        "unit": "MIO_EUR",  # Millions of euros
        "sinceTimePeriod": str(start_year),
        "untilTimePeriod": str(end_year),
    }

    # Fetch data
    data = _fetch_eurostat_data("nama_10r_3gdp", filters)

    if not data:
        logger.warning("Could not fetch GDP data from Eurostat")
        return pd.DataFrame(columns=[
            "date", "indicator", "value", "location", "location_code", "unit", "source"
        ])

    df = _parse_eurostat_json(data, "gdp_total")

    if not df.empty:
        df["unit"] = "MIO_EUR"

    return df


def fetch_employment_eurostat(
    regions: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch regional employment rate data from Eurostat.

    Uses dataset: lfst_r_lfe2emprt (Employment rates by NUTS 2 regions)

    Args:
        regions: List of region codes (default: ["ES61"] for Andalucía)
        start_year: Start year
        end_year: End year

    Returns:
        DataFrame with employment rate data

    Note: Employment rate = Employed persons / Population aged 15-64
    """
    # Employment data at NUTS3 level is limited; use NUTS2 (Andalucía)
    regions = regions or ["ES61"]
    current_year = datetime.now().year
    start_year = start_year or (current_year - 10)
    end_year = end_year or current_year

    logger.info(f"Fetching employment rate for regions: {regions}")

    filters = {
        "geo": regions,
        "age": "Y15-64",  # Working age population
        "sex": "T",       # Total (both sexes)
        "sinceTimePeriod": str(start_year),
        "untilTimePeriod": str(end_year),
    }

    data = _fetch_eurostat_data("lfst_r_lfe2emprt", filters)

    if not data:
        logger.warning("Could not fetch employment data from Eurostat")
        return pd.DataFrame(columns=[
            "date", "indicator", "value", "location", "location_code", "unit", "source"
        ])

    df = _parse_eurostat_json(data, "employment_rate")

    if not df.empty:
        df["unit"] = "percent"
        df["age_group"] = "Y15-64"
        df["sex"] = "T"

    return df


def fetch_unemployment_eurostat(
    regions: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch regional unemployment rate data from Eurostat.

    Uses dataset: lfst_r_lfu3rt (Unemployment rates by NUTS 2 regions)

    Args:
        regions: List of region codes
        start_year: Start year
        end_year: End year

    Returns:
        DataFrame with unemployment rate data

    Note: Unemployment rate = Unemployed / Active population
    """
    regions = regions or ["ES61"]
    current_year = datetime.now().year
    start_year = start_year or (current_year - 10)
    end_year = end_year or current_year

    logger.info(f"Fetching unemployment rate for regions: {regions}")

    filters = {
        "geo": regions,
        "age": "Y15-74",
        "sex": "T",
        "sinceTimePeriod": str(start_year),
        "untilTimePeriod": str(end_year),
    }

    data = _fetch_eurostat_data("lfst_r_lfu3rt", filters)

    if not data:
        logger.warning("Could not fetch unemployment data from Eurostat")
        return pd.DataFrame(columns=[
            "date", "indicator", "value", "location", "location_code", "unit", "source"
        ])

    df = _parse_eurostat_json(data, "unemployment_rate")

    if not df.empty:
        df["unit"] = "percent"
        df["age_group"] = "Y15-74"
        df["sex"] = "T"

    return df


def fetch_tourism_nights_eurostat(
    regions: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch tourism nights data from Eurostat.

    Uses dataset: tour_occ_ninat (Nights spent at tourist accommodation)

    This is an alternative to Dataestur for tourism data.
    """
    regions = regions or ["ES61"]
    current_year = datetime.now().year
    start_year = start_year or (current_year - 5)
    end_year = end_year or current_year

    logger.info(f"Fetching tourism nights for regions: {regions}")

    filters = {
        "geo": regions,
        "unit": "NR",  # Number
        "sinceTimePeriod": str(start_year),
        "untilTimePeriod": str(end_year),
    }

    data = _fetch_eurostat_data("tour_occ_ninat", filters)

    if not data:
        logger.warning("Could not fetch tourism nights from Eurostat")
        return pd.DataFrame()

    df = _parse_eurostat_json(data, "tourism_nights")

    if not df.empty:
        df["unit"] = "nights"

    return df


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Eurostat fetchers...")

    print("\n1. GDP Data (Málaga):")
    gdp_df = fetch_gdp_eurostat(regions=["ES617"], start_year=2018)
    print(f"   Records: {len(gdp_df)}")
    if not gdp_df.empty:
        print(gdp_df.tail())

    print("\n2. Employment Rate (Andalucía):")
    emp_df = fetch_employment_eurostat(regions=["ES61"], start_year=2018)
    print(f"   Records: {len(emp_df)}")
    if not emp_df.empty:
        print(emp_df.tail())

    print("\n3. Unemployment Rate (Andalucía):")
    unemp_df = fetch_unemployment_eurostat(regions=["ES61"], start_year=2018)
    print(f"   Records: {len(unemp_df)}")
    if not unemp_df.empty:
        print(unemp_df.tail())
