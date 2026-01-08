# =============================================================================
# AI Module
# =============================================================================
# This module provides AI-powered features using Anthropic's Claude:
#   - claude_client.py: API wrapper with caching and rate limiting
#   - insights.py: Anomaly detection and trend analysis
#   - summarizer.py: News and event summarization
#   - quality.py: Data quality validation
#   - reports.py: Automated report generation
#   - query.py: Natural language data queries
# =============================================================================

from .claude_client import ClaudeClient
from .insights import InsightsAgent
from .summarizer import NewsSummarizer
from .quality import DataQualityAgent
from .reports import ReportGenerator

__all__ = [
    "ClaudeClient",
    "InsightsAgent",
    "NewsSummarizer",
    "DataQualityAgent",
    "ReportGenerator",
]
