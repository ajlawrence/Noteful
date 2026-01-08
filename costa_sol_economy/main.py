#!/usr/bin/env python3
# =============================================================================
# Costa del Sol Economic Monitor - Main Pipeline
# =============================================================================
# This is the main entry point for the economic monitoring pipeline.
#
# Run modes:
#   python main.py                 # Run all due data fetches
#   python main.py --mode daily    # Run daily tasks only
#   python main.py --mode weekly   # Run weekly tasks only
#   python main.py --mode monthly  # Run monthly tasks only
#   python main.py --mode full     # Run everything regardless of schedule
#   python main.py --report        # Generate reports only (no fetching)
#   python main.py --test          # Test API connections
#
# =============================================================================

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

import pandas as pd
from dotenv import load_dotenv

# =============================================================================
# SETUP PATHS AND ENVIRONMENT
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Load environment variables from .env file
load_dotenv(SCRIPT_DIR / ".env")

# Add the script directory to Python path for imports
sys.path.insert(0, str(SCRIPT_DIR))

# =============================================================================
# IMPORTS FROM OUR MODULES
# =============================================================================

# Storage
from storage import DatabaseManager

# Fetchers
from fetchers.dataestur import (
    fetch_tourism_arrivals,
    fetch_tourism_nights,
    fetch_tourism_expenditure
)
from fetchers.eurostat import (
    fetch_gdp_eurostat,
    fetch_employment_eurostat,
    fetch_unemployment_eurostat
)
from fetchers.ine import fetch_hotel_occupancy_ine, fetch_housing_prices_ine
from fetchers.news import fetch_economic_news

# Processors
from processors.cleaner import clean_dataframe, standardize_locations
from processors.validator import validate_data
from processors.trends import (
    calculate_yoy_change,
    calculate_mom_change,
    summarize_trends
)

# AI (only import if API key is available)
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_KEY:
    from ai import ClaudeClient, InsightsAgent, NewsSummarizer, DataQualityAgent, ReportGenerator

# Outputs
from outputs.charts import ChartGenerator
from outputs.alerts import EmailAlerts

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to

    Returns:
        Configured logger
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = SCRIPT_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    return logging.getLogger("costa_sol_economy")


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class EconomyPipeline:
    """
    Main pipeline orchestrator for Costa del Sol economic data.

    This class coordinates:
    - Data fetching from multiple sources
    - Data processing and validation
    - AI-powered analysis (if configured)
    - Storage in SQLite database
    - Visualization and reporting
    - Email alerts

    Example:
        pipeline = EconomyPipeline()
        pipeline.run(mode="daily")
    """

    def __init__(
        self,
        db_path: str = "data/costa_econ.db",
        config_path: str = "config.yaml"
    ):
        """
        Initialize the pipeline.

        Args:
            db_path: Path to SQLite database
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger("costa_sol_economy.pipeline")
        self.start_time = None
        self.stats = {
            "records_fetched": 0,
            "sources_processed": 0,
            "errors": []
        }

        # Paths
        self.db_path = SCRIPT_DIR / db_path
        self.config_path = SCRIPT_DIR / config_path

        # Initialize database
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(str(self.db_path))
        self.logger.info(f"Database initialized: {self.db_path}")

        # Initialize chart generator
        self.charts = ChartGenerator(str(SCRIPT_DIR / "data" / "charts"))

        # Initialize email alerts (may not be configured)
        self.alerts = EmailAlerts()

        # Initialize AI components (if API key is available)
        self.ai_enabled = bool(ANTHROPIC_KEY)
        if self.ai_enabled:
            try:
                self.claude = ClaudeClient()
                self.insights = InsightsAgent(self.claude)
                self.summarizer = NewsSummarizer(self.claude)
                self.quality_checker = DataQualityAgent(self.claude)
                self.report_generator = ReportGenerator(self.claude)
                self.logger.info("AI features enabled")
            except Exception as e:
                self.logger.warning(f"Could not initialize AI: {e}")
                self.ai_enabled = False
        else:
            self.logger.info("AI features disabled (no ANTHROPIC_API_KEY)")

    # =========================================================================
    # MAIN RUN METHODS
    # =========================================================================

    def run(self, mode: str = "auto") -> Dict[str, Any]:
        """
        Run the pipeline.

        Args:
            mode: Run mode - "auto", "daily", "weekly", "monthly", "quarterly", "full"

        Returns:
            Dictionary with run statistics
        """
        self.start_time = datetime.now()
        self.logger.info(f"=" * 60)
        self.logger.info(f"Pipeline starting: mode={mode}")
        self.logger.info(f"=" * 60)

        try:
            # Determine which sources to run
            if mode == "full":
                self._run_all_sources()
            elif mode == "daily":
                self._run_daily()
            elif mode == "weekly":
                self._run_weekly()
            elif mode == "monthly":
                self._run_monthly()
            elif mode == "quarterly":
                self._run_quarterly()
            else:  # auto
                self._run_auto()

            # Generate charts
            self._generate_charts()

            # Send success notification
            self._send_status_alert("success")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.stats["errors"].append(str(e))
            self._send_status_alert("failed")

        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()
        self.stats["duration_seconds"] = duration
        self.stats["run_time"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info(f"Pipeline completed in {duration:.1f} seconds")
        self.logger.info(f"Records fetched: {self.stats['records_fetched']}")
        self.logger.info(f"Errors: {len(self.stats['errors'])}")

        return self.stats

    def _run_all_sources(self):
        """Run all data sources regardless of schedule."""
        self.logger.info("Running all sources...")

        # Tourism
        self._fetch_tourism_data()

        # Employment
        self._fetch_employment_data()

        # GDP
        self._fetch_gdp_data()

        # Real Estate
        self._fetch_real_estate_data()

        # News
        self._fetch_news()

    def _run_daily(self):
        """Run daily scheduled sources."""
        self.logger.info("Running daily sources...")
        self._fetch_news()

    def _run_weekly(self):
        """Run weekly scheduled sources."""
        self.logger.info("Running weekly sources...")
        self._fetch_news()

    def _run_monthly(self):
        """Run monthly scheduled sources."""
        self.logger.info("Running monthly sources...")
        self._fetch_tourism_data()
        self._fetch_real_estate_data()

    def _run_quarterly(self):
        """Run quarterly scheduled sources."""
        self.logger.info("Running quarterly sources...")
        self._fetch_employment_data()
        self._fetch_gdp_data()

    def _run_auto(self):
        """Automatically determine what needs to run based on last fetch times."""
        self.logger.info("Running in auto mode...")

        # For now, run everything - in production you'd check last fetch times
        self._run_all_sources()

    # =========================================================================
    # DATA FETCHING METHODS
    # =========================================================================

    def _fetch_tourism_data(self):
        """Fetch all tourism data sources."""
        self.logger.info("Fetching tourism data...")

        # Tourist arrivals (Dataestur)
        try:
            df = fetch_tourism_arrivals()
            if not df.empty:
                df = clean_dataframe(df)
                df = standardize_locations(df)
                df = calculate_yoy_change(df)
                self.db.save_tourism_data(df, source="dataestur")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
                self.logger.info(f"Tourist arrivals: {len(df)} records")
        except Exception as e:
            self._log_error("tourism_arrivals", e)

        # Overnight stays
        try:
            df = fetch_tourism_nights()
            if not df.empty:
                df = clean_dataframe(df)
                df = calculate_yoy_change(df)
                self.db.save_tourism_data(df, source="dataestur")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
        except Exception as e:
            self._log_error("tourism_nights", e)

        # Expenditure
        try:
            df = fetch_tourism_expenditure()
            if not df.empty:
                df = clean_dataframe(df)
                self.db.save_tourism_data(df, source="dataestur")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
        except Exception as e:
            self._log_error("tourism_expenditure", e)

        # Hotel occupancy (INE)
        try:
            df = fetch_hotel_occupancy_ine()
            if not df.empty:
                df = clean_dataframe(df)
                self.db.save_tourism_data(df, source="ine")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
        except Exception as e:
            self._log_error("hotel_occupancy", e)

    def _fetch_employment_data(self):
        """Fetch employment data from Eurostat."""
        self.logger.info("Fetching employment data...")

        # Employment rate
        try:
            df = fetch_employment_eurostat()
            if not df.empty:
                df = clean_dataframe(df)
                df = calculate_yoy_change(df)
                self.db.save_employment_data(df, source="eurostat")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
                self.logger.info(f"Employment rate: {len(df)} records")
        except Exception as e:
            self._log_error("employment_rate", e)

        # Unemployment rate
        try:
            df = fetch_unemployment_eurostat()
            if not df.empty:
                df = clean_dataframe(df)
                df = calculate_yoy_change(df)
                self.db.save_employment_data(df, source="eurostat")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
        except Exception as e:
            self._log_error("unemployment_rate", e)

    def _fetch_gdp_data(self):
        """Fetch GDP data from Eurostat."""
        self.logger.info("Fetching GDP data...")

        try:
            df = fetch_gdp_eurostat()
            if not df.empty:
                df = clean_dataframe(df)
                df = calculate_yoy_change(df)
                self.db.save_gdp_data(df, source="eurostat")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
                self.logger.info(f"GDP data: {len(df)} records")
        except Exception as e:
            self._log_error("gdp", e)

    def _fetch_real_estate_data(self):
        """Fetch real estate data from INE."""
        self.logger.info("Fetching real estate data...")

        try:
            df = fetch_housing_prices_ine()
            if not df.empty:
                df = clean_dataframe(df)
                df = calculate_yoy_change(df)
                self.db.save_real_estate_data(df, source="ine")
                self.stats["records_fetched"] += len(df)
                self.stats["sources_processed"] += 1
                self.logger.info(f"Housing prices: {len(df)} records")
        except Exception as e:
            self._log_error("housing_prices", e)

    def _fetch_news(self):
        """Fetch and process economic news."""
        self.logger.info("Fetching economic news...")

        try:
            news_df = fetch_economic_news(max_articles=20, days_back=7)
            if not news_df.empty:
                self.logger.info(f"Fetched {len(news_df)} news articles")

                # Save events to database
                for _, row in news_df.iterrows():
                    # Use AI to summarize if available
                    summary = row.get("summary", "")
                    sentiment = None
                    impact_score = None

                    if self.ai_enabled and summary:
                        try:
                            sent_result = self.summarizer.analyze_sentiment(summary)
                            sentiment = sent_result.get("sentiment")
                        except Exception:
                            pass

                    self.db.save_event(
                        date=str(row.get("date", datetime.now()))[:10],
                        title=row.get("title", "Untitled"),
                        summary=summary[:1000] if summary else None,
                        category=row.get("category"),
                        source=row.get("source"),
                        source_url=row.get("source_url"),
                        entry_type="auto",
                        sentiment=sentiment,
                        impact_score=impact_score
                    )

                self.stats["records_fetched"] += len(news_df)
                self.stats["sources_processed"] += 1
        except Exception as e:
            self._log_error("news", e)

    # =========================================================================
    # ANALYSIS & REPORTING
    # =========================================================================

    def _generate_charts(self):
        """Generate visualization charts."""
        self.logger.info("Generating charts...")

        try:
            # Get data from database
            tourism_df = self.db.get_tourism_data()
            employment_df = self.db.get_employment_data()
            gdp_df = self.db.get_gdp_data()

            # Tourism chart
            if not tourism_df.empty:
                self.charts.time_series_chart(
                    tourism_df,
                    title="Tourist Arrivals - Costa del Sol",
                    filename="tourism_arrivals.png"
                )

                if "yoy_change" in tourism_df.columns:
                    self.charts.yoy_comparison_chart(
                        tourism_df,
                        title="Tourism YoY Change",
                        filename="tourism_yoy.png"
                    )

            # Dashboard
            self.charts.create_dashboard(
                tourism_df=tourism_df,
                employment_df=employment_df,
                gdp_df=gdp_df,
                title="Costa del Sol Economic Dashboard",
                filename="dashboard.png"
            )

            self.logger.info("Charts generated successfully")

        except Exception as e:
            self._log_error("chart_generation", e)

    def generate_report(self, period: Optional[str] = None) -> str:
        """
        Generate an economic report.

        Args:
            period: Report period (e.g., "2024-01")

        Returns:
            Report content as markdown string
        """
        if not self.ai_enabled:
            return "# Report Generation\n\nAI features not available (no API key)."

        period = period or datetime.now().strftime("%Y-%m")

        try:
            tourism_df = self.db.get_tourism_data()
            employment_df = self.db.get_employment_data()
            gdp_df = self.db.get_gdp_data()
            events_df = self.db.get_events()

            report = self.report_generator.monthly_report(
                tourism_df=tourism_df,
                employment_df=employment_df,
                gdp_df=gdp_df,
                events_df=events_df,
                month=period
            )

            # Save report
            report_path = SCRIPT_DIR / "data" / f"report_{period}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report)

            self.logger.info(f"Report saved: {report_path}")

            return report

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"# Report Generation Failed\n\nError: {e}"

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _log_error(self, source: str, error: Exception):
        """Log an error and add to stats."""
        message = f"{source}: {str(error)}"
        self.logger.error(message)
        self.stats["errors"].append(message)

        # Log to database
        self.db.log_fetch(
            source_name=source,
            source_type="unknown",
            status="failed",
            error_message=str(error)
        )

    def _send_status_alert(self, status: str):
        """Send pipeline status alert."""
        if self.alerts.is_configured:
            try:
                self.alerts.send_pipeline_status(
                    status=status,
                    summary=self.stats
                )
            except Exception as e:
                self.logger.warning(f"Could not send status alert: {e}")

    def test_connections(self) -> Dict[str, bool]:
        """
        Test all API connections.

        Returns:
            Dictionary of source -> connected status
        """
        results = {}

        self.logger.info("Testing API connections...")

        # Test Eurostat
        try:
            df = fetch_gdp_eurostat(start_year=2022)
            results["eurostat"] = not df.empty
        except Exception:
            results["eurostat"] = False

        # Test INE
        try:
            df = fetch_housing_prices_ine(start_year=2022)
            results["ine"] = not df.empty
        except Exception:
            results["ine"] = False

        # Test Dataestur
        try:
            df = fetch_tourism_arrivals(start_year=2023)
            results["dataestur"] = not df.empty
        except Exception:
            results["dataestur"] = False

        # Test News RSS
        try:
            df = fetch_economic_news(max_articles=5, days_back=30)
            results["news_rss"] = not df.empty
        except Exception:
            results["news_rss"] = False

        # Test Claude (if configured)
        if self.ai_enabled:
            results["claude"] = self.claude.health_check()
        else:
            results["claude"] = None  # Not configured

        # Test Email (if configured)
        if self.alerts.is_configured:
            results["email"] = self.alerts.test_connection()
        else:
            results["email"] = None

        return results

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all stored data."""
        return self.db.get_all_data_summary()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Costa del Sol Economic Monitor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Run in auto mode
  python main.py --mode daily     # Run daily sources only
  python main.py --mode full      # Run all sources
  python main.py --report         # Generate report only
  python main.py --test           # Test API connections
  python main.py --summary        # Show data summary
        """
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "daily", "weekly", "monthly", "quarterly", "full"],
        default="auto",
        help="Run mode (default: auto)"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report only (no data fetching)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test API connections"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show data summary"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        default="data/pipeline.log",
        help="Log file path"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)

    # Initialize pipeline
    pipeline = EconomyPipeline()

    # Handle different commands
    if args.test:
        print("\n🔌 Testing API Connections...\n")
        results = pipeline.test_connections()
        for source, status in results.items():
            if status is None:
                emoji = "⚪"  # Not configured
                text = "Not configured"
            elif status:
                emoji = "✅"
                text = "Connected"
            else:
                emoji = "❌"
                text = "Failed"
            print(f"  {emoji} {source}: {text}")
        print()

    elif args.summary:
        print("\n📊 Data Summary\n")
        summary = pipeline.get_data_summary()
        for table, stats in summary.items():
            print(f"  {table}:")
            for key, value in stats.items():
                print(f"    {key}: {value}")
        print()

    elif args.report:
        print("\n📝 Generating Report...\n")
        report = pipeline.generate_report()
        print(report[:1000] + "..." if len(report) > 1000 else report)

    else:
        # Run pipeline
        print(f"\n🚀 Starting Pipeline (mode: {args.mode})...\n")
        stats = pipeline.run(mode=args.mode)
        print(f"\n✅ Pipeline completed!")
        print(f"   Records fetched: {stats['records_fetched']}")
        print(f"   Sources processed: {stats['sources_processed']}")
        print(f"   Duration: {stats.get('duration_seconds', 0):.1f}s")
        if stats['errors']:
            print(f"   Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:
                print(f"     - {error}")
        print()


# =============================================================================
# CRON / SCHEDULER INSTRUCTIONS
# =============================================================================

SCHEDULER_INSTRUCTIONS = """
# =============================================================================
# AUTOMATION SETUP
# =============================================================================

# Option 1: Cron (Linux/Mac)
# --------------------------
# Edit crontab: crontab -e
# Add these lines:

# Daily at 8:00 AM (news)
0 8 * * * cd /path/to/costa_sol_economy && python main.py --mode daily

# Monthly on the 5th at 9:00 AM
0 9 5 * * cd /path/to/costa_sol_economy && python main.py --mode monthly

# Quarterly on the 15th of Jan/Apr/Jul/Oct
0 9 15 1,4,7,10 * cd /path/to/costa_sol_economy && python main.py --mode quarterly


# Option 2: GitHub Actions
# ------------------------
# Create .github/workflows/economy-pipeline.yml:

# name: Costa Sol Economy Pipeline
# on:
#   schedule:
#     - cron: '0 8 * * *'  # Daily at 8 AM UTC
#     - cron: '0 9 5 * *'  # Monthly on 5th
#   workflow_dispatch:     # Manual trigger
#
# jobs:
#   run-pipeline:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v3
#       - uses: actions/setup-python@v4
#         with:
#           python-version: '3.10'
#       - run: pip install -r requirements.txt
#       - run: python main.py --mode auto
#         env:
#           ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}


# Option 3: Python Schedule (built-in)
# ------------------------------------
# Run: python scheduler.py
# (Creates a long-running process that handles scheduling)

"""

if __name__ == "__main__":
    main()
