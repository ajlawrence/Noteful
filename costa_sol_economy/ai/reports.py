# =============================================================================
# Report Generator
# =============================================================================
# AI-powered generation of economic reports and briefings.
#
# Features:
#   - Monthly economic summaries
#   - Trend analysis reports
#   - Executive briefings
#   - Custom report templates
# =============================================================================

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from .claude_client import ClaudeClient

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

REPORT_WRITER_PROMPT = """You are an economic analyst writing reports for Costa del Sol regional stakeholders (government officials, business owners, investors).

Your reports should:
1. Be clear and professional
2. Lead with key findings
3. Support claims with data
4. Provide actionable insights
5. Use appropriate economic terminology
6. Consider the regional context (tourism-dependent economy)

Format reports in clean Markdown with:
- Clear section headers
- Bullet points for key metrics
- Tables for data comparisons when helpful
- A forward-looking conclusion

Write in a factual, balanced tone. Acknowledge uncertainty where appropriate."""


# =============================================================================
# REPORT GENERATOR CLASS
# =============================================================================

class ReportGenerator:
    """
    AI-powered report generation for economic data.

    Example usage:
        generator = ReportGenerator()

        # Generate monthly report
        report = generator.monthly_report(
            tourism_df=tourism_data,
            employment_df=employment_data,
            gdp_df=gdp_data,
            month="2024-01"
        )

        # Generate executive brief
        brief = generator.executive_brief(
            key_metrics={"arrivals": 150000, "yoy_change": 5.2}
        )
    """

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None
    ):
        """Initialize the Report Generator."""
        self.client = claude_client or ClaudeClient()
        logger.info("ReportGenerator initialized")

    def monthly_report(
        self,
        tourism_df: Optional[pd.DataFrame] = None,
        employment_df: Optional[pd.DataFrame] = None,
        gdp_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        month: Optional[str] = None,
        include_charts: bool = False
    ) -> str:
        """
        Generate a comprehensive monthly economic report.

        Args:
            tourism_df: Tourism data DataFrame
            employment_df: Employment data DataFrame
            gdp_df: GDP data DataFrame
            events_df: Events/news DataFrame
            month: Month to report on (YYYY-MM format)
            include_charts: Whether to reference charts (assumes they exist)

        Returns:
            Markdown-formatted report string

        Example:
            report = generator.monthly_report(
                tourism_df=tourism_data,
                month="2024-06"
            )
            print(report)
        """
        month = month or datetime.now().strftime("%Y-%m")
        month_display = datetime.strptime(month + "-01", "%Y-%m-%d").strftime("%B %Y")

        # Gather data summaries
        sections = []

        sections.append(f"Report Period: {month_display}")
        sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        if tourism_df is not None and not tourism_df.empty:
            sections.append("\n## Tourism Data")
            sections.append(self._summarize_df(tourism_df, "tourism"))

        if employment_df is not None and not employment_df.empty:
            sections.append("\n## Employment Data")
            sections.append(self._summarize_df(employment_df, "employment"))

        if gdp_df is not None and not gdp_df.empty:
            sections.append("\n## GDP Data")
            sections.append(self._summarize_df(gdp_df, "gdp"))

        if events_df is not None and not events_df.empty:
            sections.append("\n## Notable Events")
            sections.append(self._summarize_events(events_df))

        data_summary = "\n".join(sections)

        prompt = f"""Write a monthly economic report for Costa del Sol based on this data:

{data_summary}

Structure the report as:
1. **Executive Summary** (3-4 sentences highlighting key findings)
2. **Tourism Sector** (if data available)
   - Performance metrics
   - Comparison to previous periods
   - Key drivers/factors
3. **Employment** (if data available)
   - Current rates
   - Trends
4. **Economic Output** (if GDP data available)
   - Growth indicators
5. **Outlook** (2-3 sentences on expectations)

Use specific numbers from the data. Keep the total report under 800 words."""

        try:
            response = self.client.send_message(
                prompt,
                system=REPORT_WRITER_PROMPT,
                model="sonnet",
                max_tokens=1500
            )

            # Add header
            report = f"# Costa del Sol Economic Report\n## {month_display}\n\n"
            report += response

            if include_charts:
                report += "\n\n---\n*Charts available in the /data/charts directory*"

            return report

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"# Report Generation Failed\n\nError: {e}"

    def executive_brief(
        self,
        key_metrics: Dict[str, Any],
        period: Optional[str] = None,
        audience: str = "general"  # "general", "investors", "government"
    ) -> str:
        """
        Generate a concise executive brief.

        Args:
            key_metrics: Dictionary of key metrics to highlight
            period: Time period for the brief
            audience: Target audience

        Returns:
            Concise executive brief string

        Example:
            brief = generator.executive_brief(
                key_metrics={
                    "tourist_arrivals": 150000,
                    "yoy_change": 5.2,
                    "hotel_occupancy": 78.5,
                    "unemployment_rate": 14.2
                },
                period="Q4 2024"
            )
        """
        period = period or datetime.now().strftime("%B %Y")

        # Format metrics for prompt
        metrics_text = "\n".join([
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in key_metrics.items()
        ])

        audience_instruction = {
            "general": "Write for a general business audience.",
            "investors": "Focus on investment implications and opportunities.",
            "government": "Emphasize policy implications and social factors."
        }.get(audience, "Write for a general business audience.")

        prompt = f"""Write a one-page executive brief on Costa del Sol's economy for {period}.

KEY METRICS:
{metrics_text}

{audience_instruction}

Format:
- **Headline** (one impactful sentence)
- **At a Glance** (3-4 bullet points with key numbers)
- **Analysis** (2-3 sentences of interpretation)
- **Outlook** (1-2 sentences)

Keep it under 250 words. Be direct and actionable."""

        try:
            response = self.client.send_message(
                prompt,
                system=REPORT_WRITER_PROMPT,
                model="sonnet",
                max_tokens=500
            )

            return f"# Executive Brief: Costa del Sol Economy\n## {period}\n\n{response}"

        except Exception as e:
            logger.error(f"Brief generation failed: {e}")
            return f"# Executive Brief\n\nGeneration failed: {e}"

    def trend_report(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        periods: int = 12
    ) -> str:
        """
        Generate a focused trend analysis report.

        Args:
            df: DataFrame with time series data
            indicator_name: Name of the indicator
            periods: Number of periods to analyze

        Returns:
            Trend analysis report string
        """
        if df.empty:
            return f"# {indicator_name} Trend Report\n\nNo data available."

        # Get recent data
        recent = df.tail(periods)
        data_summary = self._summarize_df(recent, indicator_name)

        prompt = f"""Analyze the trend in {indicator_name} for Costa del Sol:

DATA (last {periods} periods):
{data_summary}

Provide:
1. **Trend Direction**: Is it increasing, decreasing, or stable?
2. **Rate of Change**: How fast is it changing?
3. **Seasonality**: Are there seasonal patterns?
4. **Comparison**: How does recent performance compare to historical?
5. **Drivers**: What might be causing this trend?
6. **Forecast**: What might we expect in the near term?

Be specific with percentages and comparisons."""

        try:
            response = self.client.send_message(
                prompt,
                system=REPORT_WRITER_PROMPT,
                model="sonnet",
                max_tokens=800
            )

            return f"# {indicator_name} Trend Analysis\n\n{response}"

        except Exception as e:
            logger.error(f"Trend report failed: {e}")
            return f"# Trend Report\n\nGeneration failed: {e}"

    def custom_report(
        self,
        template: str,
        data: Dict[str, Any],
        title: str = "Custom Report"
    ) -> str:
        """
        Generate a report from a custom template.

        Args:
            template: Report template with placeholders
            data: Data to fill in the template
            title: Report title

        Returns:
            Generated report string

        Example:
            template = '''
            Analyze the tourism recovery in Costa del Sol.
            Focus on: {focus_areas}
            Key metrics: {metrics}
            '''
            report = generator.custom_report(
                template=template,
                data={"focus_areas": "arrivals, spending", "metrics": {...}}
            )
        """
        # Fill in template placeholders
        try:
            filled_template = template.format(**data)
        except KeyError as e:
            filled_template = template
            logger.warning(f"Template placeholder not found: {e}")

        prompt = f"""Generate a professional economic report based on this brief:

{filled_template}

Format the report with clear sections, use Markdown formatting.
Include specific data points where provided."""

        try:
            response = self.client.send_message(
                prompt,
                system=REPORT_WRITER_PROMPT,
                model="sonnet",
                max_tokens=1200
            )

            return f"# {title}\n\n{response}"

        except Exception as e:
            logger.error(f"Custom report failed: {e}")
            return f"# {title}\n\nGeneration failed: {e}"

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _summarize_df(self, df: pd.DataFrame, data_type: str) -> str:
        """Create a text summary of a DataFrame for the prompt."""
        summary = []

        summary.append(f"Records: {len(df)}")

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            if len(dates) > 0:
                summary.append(f"Period: {dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}")

        if "value" in df.columns:
            values = df["value"].dropna()
            if len(values) > 0:
                latest = values.iloc[-1]
                summary.append(f"Latest value: {latest:,.2f}")
                summary.append(f"Average: {values.mean():,.2f}")

                # Calculate simple change
                if len(values) > 1:
                    prev = values.iloc[-2]
                    if prev != 0:
                        pct_change = ((latest - prev) / prev) * 100
                        summary.append(f"Recent change: {pct_change:+.1f}%")

        if "yoy_change" in df.columns:
            yoy = df["yoy_change"].dropna()
            if len(yoy) > 0:
                summary.append(f"YoY change (latest): {yoy.iloc[-1]:+.1f}%")

        # Add a few recent data points
        if "date" in df.columns and "value" in df.columns:
            recent = df.nlargest(5, "date")[["date", "value"]].dropna()
            if not recent.empty:
                summary.append("\nRecent values:")
                for _, row in recent.iterrows():
                    date_str = pd.to_datetime(row["date"]).strftime("%Y-%m")
                    summary.append(f"  {date_str}: {row['value']:,.2f}")

        return "\n".join(summary)

    def _summarize_events(self, events_df: pd.DataFrame) -> str:
        """Summarize events data."""
        if events_df.empty:
            return "No notable events recorded."

        summary = []
        summary.append(f"Total events: {len(events_df)}")

        # Get recent events
        if "title" in events_df.columns:
            for _, row in events_df.head(5).iterrows():
                title = row.get("title", "Untitled")[:80]
                date = row.get("date", "")
                if isinstance(date, pd.Timestamp):
                    date = date.strftime("%Y-%m-%d")
                summary.append(f"- [{date}] {title}")

        return "\n".join(summary)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    print("Testing Report Generator...")

    # Create sample data
    tourism_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="M"),
        "indicator": "tourist_arrivals",
        "value": [100000, 110000, 130000, 180000, 200000, 220000],
        "yoy_change": [5.0, 7.0, 8.0, 12.0, 15.0, 10.0]
    })

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping live tests.")
    else:
        try:
            generator = ReportGenerator()

            print("\n1. Executive Brief:")
            brief = generator.executive_brief(
                key_metrics={
                    "tourist_arrivals": 220000,
                    "yoy_change_percent": 10.0,
                    "hotel_occupancy": 75.5,
                    "average_spend_per_tourist": 1250
                },
                period="June 2024"
            )
            print(brief[:500] + "...")

            print("\n2. Trend Report:")
            trend = generator.trend_report(
                tourism_df,
                "Tourist Arrivals",
                periods=6
            )
            print(trend[:500] + "...")

        except Exception as e:
            print(f"❌ Error: {e}")
