# =============================================================================
# Storage Module
# =============================================================================
# This module handles all database operations for the Costa del Sol Economic
# Monitor. It uses SQLite for simplicity and portability.
#
# Main components:
#   - models.py: Defines database table structures
#   - database.py: Functions for reading/writing data
# =============================================================================

from .models import (
    Tourism,
    Employment,
    GDP,
    RealEstate,
    Event,
    FetchLog,
    AICache,
)
from .database import DatabaseManager

# Make these available when someone does: from storage import DatabaseManager
__all__ = [
    "DatabaseManager",
    "Tourism",
    "Employment",
    "GDP",
    "RealEstate",
    "Event",
    "FetchLog",
    "AICache",
]
