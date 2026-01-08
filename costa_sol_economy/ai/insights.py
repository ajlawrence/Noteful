# =============================================================================
# Insights Agent
# =============================================================================
# AI-powered insights and anomaly detection for economic data.
#
# Features:
#   - Anomaly detection: Identify unusual data points
#   - Trend analysis: Explain what trends mean
#   - Comparative analysis: Compare current vs historical
#   - Predictions: Basic forecasting with caveats
#
# This is the PRIMARY AI feature requested by the user.
# =============================================================================

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from .claude_client import ClaudeClient

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

ANALYST_SYSTEM_PROMPT = """You are an expert economic analyst specializing in regional tourism economies, specifically the Costa del Sol (Málaga province) in Spain.

Your role is to:
1. Analyze economic data and identify meaningful patterns
2. Detect anomalies and explain their potential causes
3. Provide actionable insights for stakeholders
4. Compare current performance to historical trends

Key context about Costa del Sol:
- Primary economic driver: Tourism (beach tourism, golf, cultural tourism)
- Peak season: June-September
- Secondary peak: Easter week (Semana Santa)
- Key markets: UK, Germany, France, Nordic countries
- Regional code: ES617 (Málaga NUTS3), ES61 (Andalucía NUTS2)

When analyzing data:
- Always consider seasonality
- Note if values are nominal or real (inflation-adjusted)
- Consider external factors (economic crises, pandemics, weather events)
- Be specific with percentages and comparisons
- Acknowledge uncertainty when appropriate

Respond in a clear, professional tone. Use bullet points for key findings."""


ANOMALY_SYSTEM_PROMPT = """You are a data scientist analyzing economic time series for anomalies.

An anomaly is a data point that:
1. Deviates significantly from expected patterns
2. Cannot be explained by normal seasonality
3. May indicate data quality issues OR real economic events

For each anomaly, provide:
1. The specific value and how it differs from expected
2. Possible explanations (rank by likelihood)
3. Recommended actions (verify data, investigate further, etc.)

Be precise with numbers. If you're uncertain, say so."""


# =============================================================================
# INSIGHTS AGENT CLASS
# =============================================================================

class InsightsAgent:
    """
    AI agent for generating economic insights from data.

    This agent uses Claude to analyze economic data and provide:
    - Anomaly detection with explanations
    - Trend analysis and interpretation
    - Comparative insights (YoY, vs benchmarks)
    - Plain-language summaries for reports

    Example usage:
        agent = InsightsAgent()

        # Analyze tourism data
        insights = agent.analyze_data(tourism_df, "tourism arrivals")

        # Detect anomalies
        anomalies = agent.detect_anomalies(gdp_df, threshold=2.5)

        # Get plain-language summary
        summary = agent.summarize_trends(employment_df)
    """

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        default_model: str = "sonnet"
    ):
        """
        Initialize the Insights Agent.

        Args:
            claude_client: Existing ClaudeClient instance (or creates new one)
            default_model: Default model for analysis ("haiku", "sonnet", "opus")
        """
        self.client = claude_client or ClaudeClient()
        self.default_model = default_model
        logger.info("InsightsAgent initialized")

    # =========================================================================
    # MAIN ANALYSIS METHODS
    # =========================================================================

    def analyze_data(
        self,
        df: pd.DataFrame,
        data_description: str,
        focus_areas: Optional[List[str]] = None,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of economic data.

        Args:
            df: DataFrame with economic data
            data_description: What the data represents (e.g., "tourist arrivals")
            focus_areas: Specific aspects to focus on (e.g., ["trends", "seasonality"])
            include_recommendations: Whether to include action recommendations

        Returns:
            Dictionary with:
            - summary: High-level summary
            - key_findings: List of important findings
            - trends: Trend analysis
            - anomalies: Any detected anomalies
            - recommendations: Suggested actions (if requested)

        Example:
            result = agent.analyze_data(
                tourism_df,
                "monthly tourist arrivals to Málaga",
                focus_areas=["yoy_comparison", "seasonality"]
            )
        """
        if df.empty:
            return {
                "error": "Empty DataFrame provided",
                "summary": "No data to analyze"
            }

        # Prepare data summary for Claude
        data_summary = self._prepare_data_summary(df)

        # Build the prompt
        focus_text = ""
        if focus_areas:
            focus_text = f"\n\nFocus particularly on: {', '.join(focus_areas)}"

        prompt = f"""Analyze the following {data_description} data for Costa del Sol (Málaga):

DATA SUMMARY:
{data_summary}

Please provide:
1. A brief executive summary (2-3 sentences)
2. Key findings (3-5 bullet points)
3. Trend analysis (what direction is the data moving?)
4. Any anomalies or unusual patterns
{"5. Recommendations for stakeholders" if include_recommendations else ""}
{focus_text}

Format your response as a structured analysis."""

        try:
            response = self.client.send_message(
                prompt,
                system=ANALYST_SYSTEM_PROMPT,
                model=self.default_model,
                max_tokens=1500
            )

            # Parse the response into structured format
            result = self._parse_analysis_response(response)
            result["raw_response"] = response
            result["data_points_analyzed"] = len(df)

            logger.info(f"Analysis complete for {data_description}")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "summary": "Analysis failed due to an error"
            }

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        value_column: str = "value",
        date_column: str = "date",
        std_threshold: float = 2.5,
        explain: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect and explain anomalies in time series data.

        Uses both statistical methods and AI explanation.

        Args:
            df: DataFrame with time series data
            value_column: Column containing values
            date_column: Column containing dates
            std_threshold: Number of std deviations to consider anomaly
            explain: Whether to get AI explanation for each anomaly

        Returns:
            List of anomaly dictionaries, each containing:
            - date: Date of anomaly
            - value: The anomalous value
            - expected: What was expected
            - deviation: How far from expected (in std devs)
            - explanation: AI-generated explanation (if explain=True)

        Example:
            anomalies = agent.detect_anomalies(tourism_df, threshold=2.0)
            for a in anomalies:
                print(f"{a['date']}: {a['value']} ({a['explanation']})")
        """
        if df.empty or value_column not in df.columns:
            return []

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # Calculate statistical anomalies
        values = df[value_column].dropna()
        mean = values.mean()
        std = values.std()

        if std == 0:
            return []

        # Find anomalies
        df["_zscore"] = (df[value_column] - mean) / std
        anomaly_mask = abs(df["_zscore"]) > std_threshold

        anomalies = []

        for idx, row in df[anomaly_mask].iterrows():
            anomaly = {
                "date": str(row[date_column])[:10],
                "value": round(float(row[value_column]), 2),
                "expected": round(float(mean), 2),
                "deviation_std": round(float(row["_zscore"]), 2),
                "direction": "above" if row["_zscore"] > 0 else "below"
            }

            anomalies.append(anomaly)

        # Get AI explanations if requested
        if explain and anomalies:
            anomalies = self._explain_anomalies(df, anomalies, value_column)

        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies

    def explain_trend(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        recent_periods: int = 12
    ) -> str:
        """
        Get a plain-language explanation of trends in the data.

        Args:
            df: DataFrame with time series data
            indicator_name: Name of the indicator (e.g., "unemployment rate")
            recent_periods: How many recent periods to focus on

        Returns:
            Plain-language trend explanation

        Example:
            explanation = agent.explain_trend(
                employment_df,
                "unemployment rate",
                recent_periods=8
            )
            print(explanation)
        """
        if df.empty:
            return "No data available for trend analysis."

        # Get recent data
        recent_df = df.tail(recent_periods)
        data_summary = self._prepare_data_summary(recent_df)

        prompt = f"""Explain the trend in this {indicator_name} data for Costa del Sol in plain language.

DATA (last {recent_periods} periods):
{data_summary}

Write 2-3 sentences explaining:
1. The overall direction (increasing, decreasing, stable)
2. The rate of change
3. Any notable patterns

Use simple language that a non-expert could understand."""

        try:
            response = self.client.send_message(
                prompt,
                system=ANALYST_SYSTEM_PROMPT,
                model="haiku",  # Use cheaper model for simple task
                max_tokens=300
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Trend explanation failed: {e}")
            return f"Unable to explain trend: {e}"

    def compare_to_benchmark(
        self,
        current_value: float,
        benchmark_value: float,
        indicator_name: str,
        benchmark_name: str = "previous year"
    ) -> str:
        """
        Compare a current value to a benchmark and explain the difference.

        Args:
            current_value: Current metric value
            benchmark_value: Benchmark to compare against
            indicator_name: Name of the indicator
            benchmark_name: What the benchmark represents

        Returns:
            Explanation of the comparison

        Example:
            explanation = agent.compare_to_benchmark(
                current_value=145000,
                benchmark_value=130000,
                indicator_name="tourist arrivals",
                benchmark_name="same month last year"
            )
        """
        if benchmark_value == 0:
            return "Cannot compare: benchmark value is zero."

        pct_change = ((current_value - benchmark_value) / benchmark_value) * 100

        prompt = f"""The current {indicator_name} for Costa del Sol is {current_value:,.0f}.
Compared to {benchmark_name} ({benchmark_value:,.0f}), this is a {pct_change:+.1f}% change.

In 2-3 sentences:
1. Characterize this change (significant/modest/negligible)
2. Suggest what might explain this difference
3. Note whether this is positive or concerning for the regional economy"""

        try:
            response = self.client.send_message(
                prompt,
                system=ANALYST_SYSTEM_PROMPT,
                model="haiku",
                max_tokens=250
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return f"Current value is {pct_change:+.1f}% vs {benchmark_name}."

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare a text summary of DataFrame for Claude."""
        summary_parts = []

        # Basic stats
        summary_parts.append(f"Records: {len(df)}")

        # Date range
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dropna()
            if len(dates) > 0:
                summary_parts.append(
                    f"Period: {dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}"
                )

        # Value statistics
        if "value" in df.columns:
            values = df["value"].dropna()
            if len(values) > 0:
                summary_parts.append(f"Min: {values.min():,.2f}")
                summary_parts.append(f"Max: {values.max():,.2f}")
                summary_parts.append(f"Mean: {values.mean():,.2f}")
                summary_parts.append(f"Std Dev: {values.std():,.2f}")

        # Recent values
        if "date" in df.columns and "value" in df.columns:
            recent = df.nlargest(6, "date")[["date", "value"]]
            summary_parts.append("\nRecent values:")
            for _, row in recent.iterrows():
                date_str = pd.to_datetime(row["date"]).strftime("%Y-%m")
                summary_parts.append(f"  {date_str}: {row['value']:,.2f}")

        # YoY if available
        if "yoy_change" in df.columns:
            latest_yoy = df["yoy_change"].dropna().iloc[-1] if len(df["yoy_change"].dropna()) > 0 else None
            if latest_yoy is not None:
                summary_parts.append(f"\nLatest YoY change: {latest_yoy:+.1f}%")

        return "\n".join(summary_parts)

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's analysis response into structured format."""
        result = {
            "summary": "",
            "key_findings": [],
            "trends": "",
            "anomalies": [],
            "recommendations": []
        }

        # Simple parsing based on common patterns
        lines = response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            lower = line.lower()
            if "summary" in lower or "executive" in lower:
                current_section = "summary"
            elif "finding" in lower or "key" in lower:
                current_section = "findings"
            elif "trend" in lower:
                current_section = "trends"
            elif "anomal" in lower:
                current_section = "anomalies"
            elif "recommend" in lower:
                current_section = "recommendations"
            elif line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.")):
                # Bullet point
                content = line.lstrip("-•*0123456789. ")
                if current_section == "findings":
                    result["key_findings"].append(content)
                elif current_section == "recommendations":
                    result["recommendations"].append(content)
            elif current_section == "summary":
                result["summary"] += line + " "
            elif current_section == "trends":
                result["trends"] += line + " "

        # Clean up
        result["summary"] = result["summary"].strip()
        result["trends"] = result["trends"].strip()

        return result

    def _explain_anomalies(
        self,
        df: pd.DataFrame,
        anomalies: List[Dict],
        value_column: str
    ) -> List[Dict]:
        """Get AI explanations for detected anomalies."""
        if not anomalies:
            return anomalies

        # Prepare context
        data_summary = self._prepare_data_summary(df)

        anomaly_descriptions = []
        for a in anomalies[:5]:  # Limit to 5 to control costs
            anomaly_descriptions.append(
                f"- {a['date']}: {a['value']} ({a['direction']} expected by {abs(a['deviation_std']):.1f} std devs)"
            )

        prompt = f"""These anomalies were detected in Costa del Sol economic data:

{chr(10).join(anomaly_descriptions)}

CONTEXT:
{data_summary}

For each anomaly, provide a brief (1-2 sentence) possible explanation.
Consider: seasonality, external events (holidays, crises), data quality issues.

Format: One explanation per anomaly, in order."""

        try:
            response = self.client.send_message(
                prompt,
                system=ANOMALY_SYSTEM_PROMPT,
                model="sonnet",
                max_tokens=500
            )

            # Parse explanations and add to anomalies
            explanations = response.split("\n")
            for i, anomaly in enumerate(anomalies[:5]):
                if i < len(explanations):
                    # Clean up the explanation
                    exp = explanations[i].strip()
                    exp = exp.lstrip("-•*0123456789. ")
                    anomaly["explanation"] = exp
                else:
                    anomaly["explanation"] = "No explanation available"

            # Mark remaining anomalies as not explained
            for anomaly in anomalies[5:]:
                anomaly["explanation"] = "Not analyzed (limit reached)"

        except Exception as e:
            logger.warning(f"Could not get anomaly explanations: {e}")
            for anomaly in anomalies:
                anomaly["explanation"] = "Explanation unavailable"

        return anomalies


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    print("Testing Insights Agent...")

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=24, freq="M")
    base_values = [100, 95, 110, 140, 180, 200, 220, 210, 170, 130, 105, 100]
    values = base_values + [v * 1.05 for v in base_values]

    # Add an anomaly
    values[18] = 50  # Unexpectedly low value

    test_df = pd.DataFrame({
        "date": dates,
        "indicator": "tourist_arrivals",
        "value": values,
        "location": "Málaga"
    })

    print("\nTest Data:")
    print(test_df.head())

    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping live tests.")
    else:
        try:
            agent = InsightsAgent()

            print("\n1. Detect Anomalies:")
            anomalies = agent.detect_anomalies(test_df, std_threshold=2.0)
            for a in anomalies:
                print(f"   {a['date']}: {a['value']} - {a.get('explanation', 'No explanation')}")

            print("\n2. Explain Trend:")
            trend = agent.explain_trend(test_df, "tourist arrivals")
            print(f"   {trend}")

        except Exception as e:
            print(f"❌ Error: {e}")
