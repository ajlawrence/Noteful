# =============================================================================
# INE (Instituto Nacional de Estadística) Fetcher
# =============================================================================
# INE is Spain's national statistics institute.
# It provides detailed data on the Spanish economy, including:
#   - Hotel occupancy (Encuesta de Ocupación Hotelera - EOH)
#   - Housing prices (Índice de Precios de Vivienda - IPV)
#   - Regional accounts
#
# Data access:
#   - Some data available via JSON API
#   - Other data requires downloading Excel/CSV files
#
# Website: https://www.ine.es
# =============================================================================

import logging
import io
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

# INE API base URL
INE_API_BASE = "https://servicios.ine.es/wstempus/js"

# INE file downloads
INE_FILES_BASE = "https://www.ine.es/jaxiT3/files/t"

# Province codes
MALAGA_CODE = "29"  # Province code for Málaga
ANDALUCIA_CODE = "01"  # Autonomous community code

# Request settings
REQUEST_TIMEOUT = 60
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

@retry_decorator
def _fetch_ine_json(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """
    Fetch data from INE JSON API.

    Args:
        endpoint: API endpoint
        params: Query parameters

    Returns:
        JSON response or None
    """
    url = f"{INE_API_BASE}{endpoint}"

    try:
        logger.info(f"Fetching from INE: {url}")
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"INE API request failed: {e}")
        return None


@retry_decorator
def _download_ine_file(table_id: str, file_format: str = "csv") -> Optional[str]:
    """
    Download a data file from INE.

    Args:
        table_id: INE table identifier
        file_format: "csv" or "xlsx"

    Returns:
        File content as string, or None
    """
    # INE uses a specific URL pattern for file downloads
    # The pattern may vary; this is a common one
    url = f"{INE_FILES_BASE}/{table_id}.{file_format}"

    try:
        logger.info(f"Downloading INE file: {url}")
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text

    except requests.exceptions.RequestException as e:
        logger.error(f"INE file download failed: {e}")
        return None


def _parse_ine_period(period_str: str) -> datetime:
    """
    Parse INE period strings into datetime.

    INE uses various formats:
        "2023M12" -> December 2023
        "2023T4" -> Q4 2023
        "2023" -> Year 2023
        "Diciembre 2023" -> December 2023
    """
    try:
        # Monthly: "2023M12"
        if "M" in period_str and period_str[:4].isdigit():
            parts = period_str.split("M")
            year = int(parts[0])
            month = int(parts[1])
            return datetime(year, month, 1)

        # Quarterly: "2023T4"
        if "T" in period_str:
            parts = period_str.split("T")
            year = int(parts[0])
            quarter = int(parts[1])
            month = (quarter - 1) * 3 + 1
            return datetime(year, month, 1)

        # Annual: "2023"
        if period_str.isdigit() and len(period_str) == 4:
            return datetime(int(period_str), 1, 1)

        # Spanish month names
        spanish_months = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
        }

        for month_name, month_num in spanish_months.items():
            if month_name in period_str.lower():
                year = int("".join(filter(str.isdigit, period_str)))
                return datetime(year, month_num, 1)

        # Fallback: try pandas
        return pd.to_datetime(period_str)

    except (ValueError, TypeError):
        logger.warning(f"Could not parse INE period: {period_str}")
        return pd.NaT


# =============================================================================
# MAIN FETCHER FUNCTIONS
# =============================================================================

def fetch_hotel_occupancy_ine(
    province: str = "malaga",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch hotel occupancy data from INE.

    Uses: Encuesta de Ocupación Hotelera (EOH)
    Table: 2074 (or similar - check INE website)

    Args:
        province: "malaga" or "andalucia"
        start_year: Start year for data
        end_year: End year for data

    Returns:
        DataFrame with columns:
        - date: Period (monthly)
        - indicator: "hotel_occupancy_rate", "hotel_rooms_occupied", etc.
        - value: The value
        - location: Province/region name
        - unit: "percent" or "rooms"

    Example:
        df = fetch_hotel_occupancy_ine(province="malaga", start_year=2020)
    """
    current_year = datetime.now().year
    start_year = start_year or (current_year - 5)
    end_year = end_year or current_year

    logger.info(f"Fetching hotel occupancy for {province} ({start_year}-{end_year})")

    # Try the JSON API first
    # INE API endpoint for hotel occupancy
    # The exact operation ID may need to be looked up
    operation_id = "EOH"  # Encuesta de Ocupación Hotelera

    try:
        # INE JSON API call
        endpoint = f"/ES/DATOS_TABLA/{operation_id}"
        data = _fetch_ine_json(endpoint)

        if data and isinstance(data, list):
            records = []
            for item in data:
                # Parse based on INE response structure
                period = item.get("Periodo", item.get("T3_Periodo", ""))
                value = item.get("Valor", item.get("Data", [{}])[0].get("Valor"))
                territory = item.get("Territorio", "")

                # Filter for our province
                if province.lower() == "malaga":
                    if "málaga" not in territory.lower() and "malaga" not in territory.lower():
                        continue
                elif province.lower() == "andalucia":
                    if "andalucía" not in territory.lower() and "andalucia" not in territory.lower():
                        continue

                date = _parse_ine_period(period)
                if pd.isna(date):
                    continue

                # Filter by year
                if date.year < start_year or date.year > end_year:
                    continue

                records.append({
                    "date": date,
                    "indicator": "hotel_occupancy_rate",
                    "value": float(value) if value else None,
                    "location": territory if territory else province.title(),
                    "unit": "percent",
                    "source": "ine"
                })

            if records:
                return pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"Could not fetch hotel occupancy from INE API: {e}")

    # Fallback: return empty DataFrame with correct structure
    logger.warning("INE hotel occupancy API unavailable - returning empty DataFrame")
    logger.info("TIP: Download data manually from https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177015")

    return pd.DataFrame(columns=[
        "date", "indicator", "value", "location", "unit", "source"
    ])


def fetch_housing_prices_ine(
    region: str = "andalucia",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch housing price index from INE.

    Uses: Índice de Precios de Vivienda (IPV)
    Table: 25171

    The housing price index tracks changes in residential property prices.
    Base year: 2015 = 100

    Args:
        region: "andalucia" or "spain"
        start_year: Start year
        end_year: End year

    Returns:
        DataFrame with housing price index data

    Note: Provincial-level data (Málaga) may not be available;
          Autonomous community (Andalucía) is the most granular.
    """
    current_year = datetime.now().year
    start_year = start_year or (current_year - 10)
    end_year = end_year or current_year

    logger.info(f"Fetching housing prices for {region} ({start_year}-{end_year})")

    # Try the JSON API
    try:
        # INE operation for housing prices
        operation_id = "IPV"
        endpoint = f"/ES/DATOS_TABLA/{operation_id}"
        data = _fetch_ine_json(endpoint)

        if data and isinstance(data, list):
            records = []
            for item in data:
                period = item.get("Periodo", "")
                value = item.get("Valor")
                territory = item.get("Territorio", "")

                # Filter by region
                if region.lower() == "andalucia":
                    if "andalucía" not in territory.lower() and "andalucia" not in territory.lower():
                        continue
                elif region.lower() == "spain":
                    if "nacional" not in territory.lower() and "españa" not in territory.lower():
                        continue

                date = _parse_ine_period(period)
                if pd.isna(date):
                    continue

                if date.year < start_year or date.year > end_year:
                    continue

                records.append({
                    "date": date,
                    "indicator": "housing_price_index",
                    "value": float(value) if value else None,
                    "location": territory if territory else region.title(),
                    "unit": "index_2015_100",
                    "source": "ine"
                })

            if records:
                return pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"Could not fetch housing prices from INE API: {e}")

    logger.warning("INE housing price API unavailable - returning empty DataFrame")
    logger.info("TIP: Download data manually from https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736152838")

    return pd.DataFrame(columns=[
        "date", "indicator", "value", "location", "unit", "source"
    ])


# =============================================================================
# MANUAL CSV/EXCEL IMPORT
# =============================================================================

def import_ine_csv(
    file_path: str,
    indicator_name: str,
    date_column: str = "Periodo",
    value_column: str = "Total",
    location: str = "Málaga",
    skip_rows: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Import INE data from a manually downloaded CSV file.

    Use this when the API is unavailable or for historical data.
    Download files from: https://www.ine.es

    Args:
        file_path: Path to the CSV file
        indicator_name: Name for the indicator
        date_column: Column name containing the date/period
        value_column: Column name containing the value
        location: Location name to assign
        skip_rows: Number of header rows to skip
        encoding: File encoding (INE files are often "latin-1" or "cp1252")

    Returns:
        DataFrame in standardized format

    Example:
        df = import_ine_csv(
            "downloads/ocupacion_hotelera_malaga.csv",
            indicator_name="hotel_occupancy_rate",
            date_column="Periodo",
            value_column="Grado de ocupación"
        )
    """
    try:
        # Try different encodings
        for enc in [encoding, "latin-1", "cp1252", "utf-8-sig"]:
            try:
                raw_df = pd.read_csv(
                    file_path,
                    skiprows=skip_rows,
                    encoding=enc,
                    sep=";",  # INE often uses semicolon
                    decimal=","  # Spanish decimal separator
                )
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            raise ValueError("Could not read file with any encoding")

        # Find the right columns (INE column names vary)
        if date_column not in raw_df.columns:
            # Try to find a period-like column
            for col in raw_df.columns:
                if "periodo" in col.lower() or "fecha" in col.lower():
                    date_column = col
                    break

        if value_column not in raw_df.columns:
            # Try to find a value-like column
            for col in raw_df.columns:
                if "total" in col.lower() or "valor" in col.lower():
                    value_column = col
                    break

        # Create standardized DataFrame
        df = pd.DataFrame({
            "date": raw_df[date_column].apply(_parse_ine_period),
            "indicator": indicator_name,
            "value": pd.to_numeric(
                raw_df[value_column].astype(str).str.replace(",", ".").str.replace(" ", ""),
                errors="coerce"
            ),
            "location": location,
            "source": "ine_csv"
        })

        df = df.dropna(subset=["date", "value"])

        logger.info(f"Imported {len(df)} records from {file_path}")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error importing INE CSV: {e}")
        return pd.DataFrame()


def import_ine_excel(
    file_path: str,
    indicator_name: str,
    sheet_name: int = 0,
    date_column: str = "Periodo",
    value_column: str = "Total",
    location: str = "Málaga",
    skip_rows: int = 0
) -> pd.DataFrame:
    """
    Import INE data from a manually downloaded Excel file.

    Args:
        file_path: Path to the Excel file
        indicator_name: Name for the indicator
        sheet_name: Sheet index or name
        date_column: Column name containing the date
        value_column: Column name containing the value
        location: Location name
        skip_rows: Rows to skip

    Returns:
        DataFrame in standardized format
    """
    try:
        raw_df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            skiprows=skip_rows
        )

        df = pd.DataFrame({
            "date": raw_df[date_column].apply(_parse_ine_period),
            "indicator": indicator_name,
            "value": pd.to_numeric(raw_df[value_column], errors="coerce"),
            "location": location,
            "source": "ine_excel"
        })

        df = df.dropna(subset=["date", "value"])

        logger.info(f"Imported {len(df)} records from {file_path}")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error importing INE Excel: {e}")
        return pd.DataFrame()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing INE fetchers...")

    print("\n1. Hotel Occupancy (Málaga):")
    hotel_df = fetch_hotel_occupancy_ine(province="malaga", start_year=2022)
    print(f"   Records: {len(hotel_df)}")
    if not hotel_df.empty:
        print(hotel_df.head())

    print("\n2. Housing Prices (Andalucía):")
    housing_df = fetch_housing_prices_ine(region="andalucia", start_year=2020)
    print(f"   Records: {len(housing_df)}")
    if not housing_df.empty:
        print(housing_df.head())
