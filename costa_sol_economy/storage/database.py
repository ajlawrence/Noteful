# =============================================================================
# Database Manager
# =============================================================================
# This file provides easy-to-use functions for reading and writing data
# to the SQLite database. Think of it as the "data access layer".
#
# MAIN CLASS: DatabaseManager
#   - save_tourism_data(df)     -> Save tourism DataFrame to database
#   - save_employment_data(df)  -> Save employment DataFrame to database
#   - get_tourism_data(...)     -> Read tourism data from database
#   - ... and more
#
# =============================================================================

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from .models import (
    Base,
    Tourism,
    Employment,
    GDP,
    RealEstate,
    Event,
    FetchLog,
    AICache,
    create_all_tables,
)

# Set up logging (you'll see these messages in the console/log file)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all database operations for the Costa del Sol Economic Monitor.

    This class provides a simple interface to:
    - Save data from API fetches
    - Retrieve data for analysis
    - Log fetch operations
    - Cache AI responses

    Example usage:
        db = DatabaseManager("data/costa_econ.db")
        db.save_tourism_data(my_dataframe)
        tourism_df = db.get_tourism_data(indicator="arrivals")
    """

    def __init__(self, database_path: str = "data/costa_econ.db"):
        """
        Initialize the database connection.

        Args:
            database_path: Path to the SQLite database file.
                          Will be created if it doesn't exist.
        """
        self.database_path = database_path

        # Create the database engine (connection pool)
        # echo=False means don't print every SQL query (set to True for debugging)
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            echo=False,
            connect_args={"check_same_thread": False}  # Required for SQLite
        )

        # Create a session factory (sessions are like database transactions)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create all tables if they don't exist
        create_all_tables(database_path)

        logger.info(f"Database initialized at: {database_path}")

    # =========================================================================
    # CONTEXT MANAGER (for 'with' statements)
    # =========================================================================

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # =========================================================================
    # TOURISM DATA OPERATIONS
    # =========================================================================

    def save_tourism_data(
        self,
        df: pd.DataFrame,
        source: str,
        source_url: Optional[str] = None
    ) -> int:
        """
        Save tourism data from a DataFrame to the database.

        Args:
            df: DataFrame with columns: date, indicator, value, location (optional)
            source: Data source name (e.g., "dataestur", "ine")
            source_url: URL where data was fetched from

        Returns:
            Number of records saved

        Example:
            df = pd.DataFrame({
                'date': ['2024-01-01', '2024-02-01'],
                'indicator': ['arrivals', 'arrivals'],
                'value': [125000, 130000]
            })
            db.save_tourism_data(df, source="dataestur")
        """
        session = self.get_session()
        records_saved = 0

        try:
            for _, row in df.iterrows():
                # Create a new Tourism record
                record = Tourism(
                    date=pd.to_datetime(row['date']).date(),
                    indicator=row['indicator'],
                    value=row.get('value'),
                    location=row.get('location', 'Málaga'),
                    breakdown_type=row.get('breakdown_type'),
                    breakdown_value=row.get('breakdown_value'),
                    unit=row.get('unit'),
                    source=source,
                    source_url=source_url,
                    fetched_at=datetime.utcnow()
                )
                session.add(record)
                records_saved += 1

            # Commit all changes to the database
            session.commit()
            logger.info(f"Saved {records_saved} tourism records from {source}")

        except Exception as e:
            # If something goes wrong, undo all changes
            session.rollback()
            logger.error(f"Error saving tourism data: {e}")
            raise

        finally:
            session.close()

        return records_saved

    def get_tourism_data(
        self,
        indicator: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        location: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve tourism data from the database.

        Args:
            indicator: Filter by indicator type (e.g., "arrivals")
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            location: Filter by location

        Returns:
            DataFrame with tourism data

        Example:
            df = db.get_tourism_data(
                indicator="arrivals",
                start_date="2023-01-01",
                end_date="2024-01-01"
            )
        """
        session = self.get_session()

        try:
            # Start building the query
            query = session.query(Tourism)

            # Apply filters if provided
            if indicator:
                query = query.filter(Tourism.indicator == indicator)
            if start_date:
                query = query.filter(Tourism.date >= start_date)
            if end_date:
                query = query.filter(Tourism.date <= end_date)
            if location:
                query = query.filter(Tourism.location == location)

            # Order by date
            query = query.order_by(Tourism.date)

            # Convert to DataFrame
            records = query.all()
            if not records:
                return pd.DataFrame()

            data = [{
                'id': r.id,
                'date': r.date,
                'indicator': r.indicator,
                'value': r.value,
                'location': r.location,
                'breakdown_type': r.breakdown_type,
                'breakdown_value': r.breakdown_value,
                'unit': r.unit,
                'source': r.source,
                'yoy_change': r.yoy_change,
                'mom_change': r.mom_change
            } for r in records]

            return pd.DataFrame(data)

        finally:
            session.close()

    # =========================================================================
    # EMPLOYMENT DATA OPERATIONS
    # =========================================================================

    def save_employment_data(
        self,
        df: pd.DataFrame,
        source: str,
        source_url: Optional[str] = None
    ) -> int:
        """Save employment data from a DataFrame to the database."""
        session = self.get_session()
        records_saved = 0

        try:
            for _, row in df.iterrows():
                record = Employment(
                    date=pd.to_datetime(row['date']).date(),
                    indicator=row['indicator'],
                    value=row.get('value'),
                    location=row.get('location', 'Málaga'),
                    age_group=row.get('age_group'),
                    sex=row.get('sex'),
                    unit=row.get('unit'),
                    source=source,
                    source_url=source_url,
                    fetched_at=datetime.utcnow()
                )
                session.add(record)
                records_saved += 1

            session.commit()
            logger.info(f"Saved {records_saved} employment records from {source}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving employment data: {e}")
            raise

        finally:
            session.close()

        return records_saved

    def get_employment_data(
        self,
        indicator: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        location: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve employment data from the database."""
        session = self.get_session()

        try:
            query = session.query(Employment)

            if indicator:
                query = query.filter(Employment.indicator == indicator)
            if start_date:
                query = query.filter(Employment.date >= start_date)
            if end_date:
                query = query.filter(Employment.date <= end_date)
            if location:
                query = query.filter(Employment.location == location)

            query = query.order_by(Employment.date)

            records = query.all()
            if not records:
                return pd.DataFrame()

            data = [{
                'id': r.id,
                'date': r.date,
                'indicator': r.indicator,
                'value': r.value,
                'location': r.location,
                'age_group': r.age_group,
                'sex': r.sex,
                'unit': r.unit,
                'source': r.source,
                'yoy_change': r.yoy_change,
                'qoq_change': r.qoq_change
            } for r in records]

            return pd.DataFrame(data)

        finally:
            session.close()

    # =========================================================================
    # GDP DATA OPERATIONS
    # =========================================================================

    def save_gdp_data(
        self,
        df: pd.DataFrame,
        source: str,
        source_url: Optional[str] = None
    ) -> int:
        """Save GDP data from a DataFrame to the database."""
        session = self.get_session()
        records_saved = 0

        try:
            for _, row in df.iterrows():
                record = GDP(
                    date=pd.to_datetime(row['date']).date(),
                    indicator=row['indicator'],
                    value=row.get('value'),
                    location=row.get('location', 'Málaga'),
                    sector=row.get('sector'),
                    sector_code=row.get('sector_code'),
                    unit=row.get('unit'),
                    source=source,
                    source_url=source_url,
                    fetched_at=datetime.utcnow()
                )
                session.add(record)
                records_saved += 1

            session.commit()
            logger.info(f"Saved {records_saved} GDP records from {source}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving GDP data: {e}")
            raise

        finally:
            session.close()

        return records_saved

    def get_gdp_data(
        self,
        indicator: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve GDP data from the database."""
        session = self.get_session()

        try:
            query = session.query(GDP)

            if indicator:
                query = query.filter(GDP.indicator == indicator)
            if start_date:
                query = query.filter(GDP.date >= start_date)
            if end_date:
                query = query.filter(GDP.date <= end_date)

            query = query.order_by(GDP.date)

            records = query.all()
            if not records:
                return pd.DataFrame()

            data = [{
                'id': r.id,
                'date': r.date,
                'indicator': r.indicator,
                'value': r.value,
                'location': r.location,
                'sector': r.sector,
                'sector_code': r.sector_code,
                'unit': r.unit,
                'source': r.source,
                'yoy_change': r.yoy_change,
                'growth_rate': r.growth_rate
            } for r in records]

            return pd.DataFrame(data)

        finally:
            session.close()

    # =========================================================================
    # REAL ESTATE DATA OPERATIONS
    # =========================================================================

    def save_real_estate_data(
        self,
        df: pd.DataFrame,
        source: str,
        source_url: Optional[str] = None
    ) -> int:
        """Save real estate data from a DataFrame to the database."""
        session = self.get_session()
        records_saved = 0

        try:
            for _, row in df.iterrows():
                record = RealEstate(
                    date=pd.to_datetime(row['date']).date(),
                    indicator=row['indicator'],
                    value=row.get('value'),
                    location=row.get('location', 'Andalucía'),
                    property_type=row.get('property_type'),
                    unit=row.get('unit'),
                    source=source,
                    source_url=source_url,
                    fetched_at=datetime.utcnow()
                )
                session.add(record)
                records_saved += 1

            session.commit()
            logger.info(f"Saved {records_saved} real estate records from {source}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving real estate data: {e}")
            raise

        finally:
            session.close()

        return records_saved

    # =========================================================================
    # EVENTS DATA OPERATIONS
    # =========================================================================

    def save_event(
        self,
        date: str,
        title: str,
        summary: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        source_url: Optional[str] = None,
        entry_type: str = "auto",
        sentiment: Optional[str] = None,
        impact_score: Optional[float] = None
    ) -> int:
        """
        Save a single event/news item to the database.

        Args:
            date: Event date (YYYY-MM-DD)
            title: Event title
            summary: AI-generated or manual summary
            category: Category (e.g., "tourism", "infrastructure")
            source: Source name
            source_url: URL to original article
            entry_type: "auto" or "manual"
            sentiment: "positive", "negative", or "neutral"
            impact_score: Estimated impact (0-1)

        Returns:
            ID of the saved event
        """
        session = self.get_session()

        try:
            record = Event(
                date=pd.to_datetime(date).date(),
                title=title,
                summary=summary,
                category=category,
                source=source,
                source_url=source_url,
                entry_type=entry_type,
                sentiment=sentiment,
                impact_score=impact_score,
                fetched_at=datetime.utcnow()
            )
            session.add(record)
            session.commit()
            logger.info(f"Saved event: {title[:50]}...")
            return record.id

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving event: {e}")
            raise

        finally:
            session.close()

    def get_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve events from the database."""
        session = self.get_session()

        try:
            query = session.query(Event)

            if start_date:
                query = query.filter(Event.date >= start_date)
            if end_date:
                query = query.filter(Event.date <= end_date)
            if category:
                query = query.filter(Event.category == category)

            query = query.order_by(Event.date.desc())

            records = query.all()
            if not records:
                return pd.DataFrame()

            data = [{
                'id': r.id,
                'date': r.date,
                'title': r.title,
                'summary': r.summary,
                'category': r.category,
                'sentiment': r.sentiment,
                'impact_score': r.impact_score,
                'source': r.source,
                'source_url': r.source_url,
                'entry_type': r.entry_type
            } for r in records]

            return pd.DataFrame(data)

        finally:
            session.close()

    # =========================================================================
    # FETCH LOG OPERATIONS
    # =========================================================================

    def log_fetch(
        self,
        source_name: str,
        source_type: str,
        status: str,
        endpoint: Optional[str] = None,
        records_fetched: Optional[int] = None,
        error_message: Optional[str] = None,
        response_code: Optional[int] = None,
        started_at: Optional[datetime] = None
    ) -> int:
        """
        Log a data fetch attempt.

        Args:
            source_name: Name of the data source
            source_type: Type of source (e.g., "dataestur", "eurostat")
            status: "success", "failed", or "partial"
            endpoint: API endpoint or URL
            records_fetched: Number of records fetched
            error_message: Error message if failed
            response_code: HTTP response code
            started_at: When the fetch started

        Returns:
            ID of the log entry
        """
        session = self.get_session()

        try:
            record = FetchLog(
                source_name=source_name,
                source_type=source_type,
                status=status,
                endpoint=endpoint,
                started_at=started_at or datetime.utcnow(),
                completed_at=datetime.utcnow(),
                records_fetched=records_fetched,
                error_message=error_message,
                response_code=response_code
            )
            session.add(record)
            session.commit()
            return record.id

        except Exception as e:
            session.rollback()
            logger.error(f"Error logging fetch: {e}")
            raise

        finally:
            session.close()

    def get_last_successful_fetch(self, source_name: str) -> Optional[datetime]:
        """
        Get the timestamp of the last successful fetch for a source.

        Useful for determining if we need to fetch new data.
        """
        session = self.get_session()

        try:
            record = (
                session.query(FetchLog)
                .filter(FetchLog.source_name == source_name)
                .filter(FetchLog.status == "success")
                .order_by(FetchLog.completed_at.desc())
                .first()
            )
            return record.completed_at if record else None

        finally:
            session.close()

    # =========================================================================
    # AI CACHE OPERATIONS
    # =========================================================================

    def get_cached_ai_response(self, prompt: str, model: str) -> Optional[str]:
        """
        Check if we have a cached AI response for this prompt.

        Args:
            prompt: The prompt that was sent to the AI
            model: The model that was used

        Returns:
            Cached response if found and not expired, None otherwise
        """
        # Create a hash of the prompt + model to look up
        request_hash = hashlib.sha256(
            f"{model}:{prompt}".encode()
        ).hexdigest()

        session = self.get_session()

        try:
            record = (
                session.query(AICache)
                .filter(AICache.request_hash == request_hash)
                .first()
            )

            if record:
                # Check if expired
                if record.expires_at and record.expires_at < datetime.utcnow():
                    # Expired - delete it
                    session.delete(record)
                    session.commit()
                    return None

                # Update hit count
                record.hit_count += 1
                record.last_hit_at = datetime.utcnow()
                session.commit()

                logger.debug(f"AI cache hit for hash {request_hash[:16]}...")
                return record.response

            return None

        finally:
            session.close()

    def cache_ai_response(
        self,
        prompt: str,
        model: str,
        response: str,
        tokens_used: Optional[int] = None,
        expiry_hours: int = 168  # 1 week default
    ) -> None:
        """
        Cache an AI response for future use.

        Args:
            prompt: The prompt that was sent
            model: The model that was used
            response: The response to cache
            tokens_used: Number of tokens used
            expiry_hours: How long to keep the cache (default 1 week)
        """
        request_hash = hashlib.sha256(
            f"{model}:{prompt}".encode()
        ).hexdigest()

        session = self.get_session()

        try:
            record = AICache(
                request_hash=request_hash,
                prompt_preview=prompt[:500],
                model_used=model,
                response=response,
                tokens_used=tokens_used,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            session.add(record)
            session.commit()
            logger.debug(f"Cached AI response with hash {request_hash[:16]}...")

        except Exception as e:
            session.rollback()
            # Might fail if duplicate - that's okay
            logger.debug(f"Could not cache AI response: {e}")

        finally:
            session.close()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_all_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all data in the database.

        Returns:
            Dictionary with counts and date ranges for each table.
        """
        session = self.get_session()

        try:
            summary = {}

            # Tourism
            tourism_count = session.query(Tourism).count()
            summary['tourism'] = {
                'count': tourism_count,
                'indicators': session.query(Tourism.indicator).distinct().count()
            }

            # Employment
            employment_count = session.query(Employment).count()
            summary['employment'] = {
                'count': employment_count,
                'indicators': session.query(Employment.indicator).distinct().count()
            }

            # GDP
            gdp_count = session.query(GDP).count()
            summary['gdp'] = {
                'count': gdp_count,
                'indicators': session.query(GDP.indicator).distinct().count()
            }

            # Events
            events_count = session.query(Event).count()
            summary['events'] = {
                'count': events_count,
                'categories': session.query(Event.category).distinct().count()
            }

            # Fetch logs
            summary['fetch_logs'] = {
                'total': session.query(FetchLog).count(),
                'successful': session.query(FetchLog).filter(
                    FetchLog.status == 'success'
                ).count()
            }

            return summary

        finally:
            session.close()

    def execute_raw_query(self, query: str) -> pd.DataFrame:
        """
        Execute a raw SQL query and return results as DataFrame.

        USE WITH CAUTION - this bypasses the ORM.

        Args:
            query: SQL query string

        Returns:
            DataFrame with query results
        """
        with self.engine.connect() as conn:
            result = pd.read_sql(text(query), conn)
        return result
