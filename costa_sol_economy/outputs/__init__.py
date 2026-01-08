# =============================================================================
# Outputs Module
# =============================================================================
# This module handles all output generation:
#   - charts.py: Matplotlib visualizations
#   - alerts.py: Email notifications
# =============================================================================

from .charts import ChartGenerator
from .alerts import EmailAlerts

__all__ = ["ChartGenerator", "EmailAlerts"]
