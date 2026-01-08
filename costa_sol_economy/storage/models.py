# =============================================================================
# Database Models
# =============================================================================
# This file defines the structure of all database tables using SQLAlchemy.
# Think of each class below as a "blueprint" for a database table.
#
# HOW TO READ THIS FILE:
# - Each class = one database table
# - Each variable inside a class = one column in that table
# - Column() defines what type of data goes in that column
#
# EXAMPLE:
#   class Tourism:
#       date = Column(Date)        # This creates a column called "date"
#       value = Column(Float)      # This creates a column called "value"
#
# =============================================================================

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Date,
    DateTime,
    Text,
    Boolean,
    create_engine,
)
from sqlalchemy.orm import declarative_base

# This creates a "base" class that all our table classes will inherit from
# Don't worry about the details - just know that all tables need this
Base = declarative_base()


# =============================================================================
# TOURISM TABLE
# =============================================================================
# Stores all tourism-related data: arrivals, overnight stays, expenditure, etc.
# =============================================================================

class Tourism(Base):
    """
    Tourism indicators for Costa del Sol / Málaga region.

    Example row:
        date: 2024-01-01
        indicator: "tourist_arrivals"
        value: 125000
        location: "Málaga"
        source: "dataestur"
    """

    # This is the name of the table in the SQLite database
    __tablename__ = "tourism"

    # ----- COLUMNS -----

    # Every table needs a unique ID for each row (auto-increments: 1, 2, 3, ...)
    id = Column(Integer, primary_key=True, autoincrement=True)

    # The date this data point refers to (e.g., January 2024)
    date = Column(Date, nullable=False, index=True)

    # What type of data this is (e.g., "arrivals", "nights", "expenditure")
    indicator = Column(String(100), nullable=False, index=True)

    # The actual numeric value
    value = Column(Float, nullable=True)

    # Geographic location (e.g., "Málaga", "Andalucía", "Costa del Sol")
    location = Column(String(100), nullable=False, default="Málaga")

    # Additional breakdown (e.g., country of origin, accommodation type)
    breakdown_type = Column(String(50), nullable=True)  # e.g., "by_country"
    breakdown_value = Column(String(100), nullable=True)  # e.g., "United Kingdom"

    # Unit of measurement (e.g., "persons", "nights", "EUR")
    unit = Column(String(50), nullable=True)

    # Where this data came from
    source = Column(String(50), nullable=False)  # e.g., "dataestur", "ine"
    source_url = Column(String(500), nullable=True)

    # When we fetched this data
    fetched_at = Column(DateTime, default=datetime.utcnow)

    # Calculated fields (filled in by processors)
    yoy_change = Column(Float, nullable=True)  # Year-over-year % change
    mom_change = Column(Float, nullable=True)  # Month-over-month % change

    def __repr__(self):
        """How this object appears when printed (for debugging)."""
        return f"<Tourism {self.date} | {self.indicator}: {self.value}>"


# =============================================================================
# EMPLOYMENT TABLE
# =============================================================================
# Stores employment and unemployment data for the region.
# =============================================================================

class Employment(Base):
    """
    Employment indicators for Málaga / Andalucía region.

    Example row:
        date: 2024-01-01
        indicator: "unemployment_rate"
        value: 15.2
        location: "Málaga"
        source: "eurostat"
    """

    __tablename__ = "employment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    indicator = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=True)
    location = Column(String(100), nullable=False, default="Málaga")

    # Demographic breakdowns
    age_group = Column(String(20), nullable=True)  # e.g., "Y15-64"
    sex = Column(String(10), nullable=True)  # "M", "F", "T" (total)

    unit = Column(String(50), nullable=True)  # e.g., "percent", "thousands"
    source = Column(String(50), nullable=False)
    source_url = Column(String(500), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    # Calculated fields
    yoy_change = Column(Float, nullable=True)
    qoq_change = Column(Float, nullable=True)  # Quarter-over-quarter

    def __repr__(self):
        return f"<Employment {self.date} | {self.indicator}: {self.value}>"


# =============================================================================
# GDP TABLE
# =============================================================================
# Stores Gross Domestic Product and economic output data.
# =============================================================================

class GDP(Base):
    """
    GDP and economic output indicators for Málaga region.

    Example row:
        date: 2023-01-01
        indicator: "gdp_total"
        value: 25430.5
        unit: "MIO_EUR" (millions of euros)
    """

    __tablename__ = "gdp"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    indicator = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=True)
    location = Column(String(100), nullable=False, default="Málaga")

    # Economic sector (for GVA breakdown)
    sector = Column(String(100), nullable=True)  # e.g., "Accommodation", "Total"
    sector_code = Column(String(20), nullable=True)  # e.g., "I", "G-I"

    unit = Column(String(50), nullable=True)
    source = Column(String(50), nullable=False)
    source_url = Column(String(500), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    # Calculated fields
    yoy_change = Column(Float, nullable=True)
    growth_rate = Column(Float, nullable=True)

    def __repr__(self):
        return f"<GDP {self.date} | {self.indicator}: {self.value}>"


# =============================================================================
# REAL ESTATE TABLE
# =============================================================================
# Stores housing and property market data.
# =============================================================================

class RealEstate(Base):
    """
    Real estate and housing indicators.

    Example row:
        date: 2024-01-01
        indicator: "housing_price_index"
        value: 142.3
        location: "Andalucía"
    """

    __tablename__ = "real_estate"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    indicator = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=True)
    location = Column(String(100), nullable=False, default="Andalucía")

    # Property type breakdown
    property_type = Column(String(50), nullable=True)  # e.g., "new", "second_hand"

    unit = Column(String(50), nullable=True)
    source = Column(String(50), nullable=False)
    source_url = Column(String(500), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    # Calculated fields
    yoy_change = Column(Float, nullable=True)
    qoq_change = Column(Float, nullable=True)

    def __repr__(self):
        return f"<RealEstate {self.date} | {self.indicator}: {self.value}>"


# =============================================================================
# EVENTS TABLE
# =============================================================================
# Stores qualitative events: news summaries, manual entries, notable occurrences.
# =============================================================================

class Event(Base):
    """
    Qualitative events and news that may impact the economy.

    Example row:
        date: 2024-06-15
        title: "New airport terminal opens in Málaga"
        summary: "The T3 terminal expansion increases capacity by 30%..."
        category: "infrastructure"
        source: "Diario Sur"
    """

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)

    # Event details
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)  # AI-generated or manual summary
    full_text = Column(Text, nullable=True)  # Original article text (optional)

    # Classification
    category = Column(String(50), nullable=True)  # e.g., "tourism", "infrastructure"
    sentiment = Column(String(20), nullable=True)  # "positive", "negative", "neutral"
    impact_score = Column(Float, nullable=True)  # AI-estimated impact (0-1)

    # Source information
    source = Column(String(100), nullable=True)
    source_url = Column(String(500), nullable=True)

    # Entry metadata
    entry_type = Column(String(20), nullable=False, default="auto")  # "auto" or "manual"
    fetched_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Event {self.date} | {self.title[:50]}...>"


# =============================================================================
# FETCH LOG TABLE
# =============================================================================
# Tracks all data fetch attempts (for monitoring and debugging).
# =============================================================================

class FetchLog(Base):
    """
    Log of all data fetching attempts.

    Used for:
    - Monitoring pipeline health
    - Debugging failed fetches
    - Avoiding duplicate fetches
    """

    __tablename__ = "fetch_log"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # What was fetched
    source_name = Column(String(100), nullable=False, index=True)
    source_type = Column(String(50), nullable=False)  # e.g., "dataestur", "eurostat"
    endpoint = Column(String(500), nullable=True)

    # When it was fetched
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Result
    status = Column(String(20), nullable=False)  # "success", "failed", "partial"
    records_fetched = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # Response metadata
    response_code = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<FetchLog {self.source_name} | {self.status} @ {self.started_at}>"


# =============================================================================
# AI CACHE TABLE
# =============================================================================
# Caches AI/LLM responses to save money and reduce latency.
# =============================================================================

class AICache(Base):
    """
    Cache for AI/LLM responses.

    Why cache?
    - Claude API calls cost money
    - Same analysis on same data = same result
    - Faster subsequent runs
    """

    __tablename__ = "ai_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # What was the request
    request_hash = Column(String(64), nullable=False, unique=True, index=True)
    prompt_preview = Column(String(500), nullable=True)  # First 500 chars
    model_used = Column(String(50), nullable=False)

    # The cached response
    response = Column(Text, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    tokens_used = Column(Integer, nullable=True)

    # Was this cache hit used?
    hit_count = Column(Integer, default=0)
    last_hit_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<AICache {self.request_hash[:16]}... | hits: {self.hit_count}>"


# =============================================================================
# HELPER FUNCTION: Create all tables
# =============================================================================

def create_all_tables(database_path: str) -> None:
    """
    Create all database tables if they don't exist.

    This function is called when you first run the pipeline.
    It's safe to call multiple times - it won't delete existing data.

    Args:
        database_path: Path to the SQLite database file
                      Example: "data/costa_econ.db"
    """
    # Create a connection to the database
    engine = create_engine(f"sqlite:///{database_path}")

    # Create all tables defined above
    Base.metadata.create_all(engine)

    print(f"✅ Database tables created/verified at: {database_path}")
