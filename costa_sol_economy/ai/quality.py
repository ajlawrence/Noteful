# =============================================================================
# Data Quality Agent
# =============================================================================
# AI-powered data quality validation and suggestions.
#
# Features:
#   - Validate data for potential issues
#   - Suggest corrections for anomalies
#   - Explain data quality problems
#   - Provide data cleaning recommendations
# =============================================================================

import logging
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from .claude_client import ClaudeClient

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

DATA_QUALITY_PROMPT = """You are a data quality specialist analyzing economic data for the Costa del Sol region.

Your role is to:
1. Identify potential data quality issues
2. Distinguish between real anomalies and data errors
3. Suggest corrections or flag for manual review
4. Provide confidence levels for your assessments

When analyzing data quality, consider:
- Expected ranges for economic indicators
- Seasonal patterns (tourism peaks in summer, dips in winter)
- Year-over-year consistency
- Common data entry errors (decimal places, unit mismatches)
- Missing data patterns

Be specific and provide actionable recommendations."""


# =============================================================================
# DATA QUALITY AGENT CLASS
# =============================================================================

class DataQualityAgent:
    """
    AI agent for data quality validation.

    Uses Claude to:
    - Analyze data for quality issues
    - Distinguish real anomalies from errors
    - Suggest corrections
    - Explain quality problems

    Example usage:
        agent = DataQualityAgent()

        # Validate a dataset
        issues = agent.validate_quality(tourism_df)

        # Check a specific value
        check = agent.check_value(
            value=50000,
            indicator="tourist_arrivals",
            context="January data, typically low season"
        )
    """

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None
    ):
        """Initialize the Data Quality Agent."""
        self.client = claude_client or ClaudeClient()
        logger.info("DataQualityAgent initialized")

    def validate_quality(
        self,
        df: pd.DataFrame,
        indicator_name: str = "economic data",
        check_level: str = "standard"  # "quick", "standard", "thorough"
    ) -> Dict[str, Any]:
        """
        Validate data quality using AI analysis.

        Args:
            df: DataFrame to validate
            indicator_name: Description of what the data represents
            check_level: How thorough to be ("quick", "standard", "thorough")

        Returns:
            Dictionary with:
            - is_valid: Overall quality assessment (True/False)
            - confidence: Confidence in assessment (0-1)
            - issues: List of identified issues
            - recommendations: List of recommendations
            - summary: Plain-language summary

        Example:
            result = agent.validate_quality(tourism_df, "tourist arrivals")
            if not result["is_valid"]:
                print("Quality issues found:")
                for issue in result["issues"]:
                    print(f"  - {issue}")
        """
        if df.empty:
            return {
                "is_valid": False,
                "confidence": 1.0,
                "issues": ["DataFrame is empty"],
                "recommendations": ["Verify data source"],
                "summary": "No data to validate."
            }

        # Prepare data summary
        summary = self._prepare_quality_summary(df)

        # Adjust detail level based on check_level
        detail_instruction = {
            "quick": "Focus on obvious issues only. Be brief.",
            "standard": "Identify main quality concerns. Provide moderate detail.",
            "thorough": "Perform comprehensive analysis. Check all aspects."
        }.get(check_level, "Provide standard level of detail.")

        prompt = f"""Analyze the quality of this {indicator_name} data for Costa del Sol:

DATA SUMMARY:
{summary}

{detail_instruction}

Assess:
1. Missing data patterns - are there unexplained gaps?
2. Value ranges - are values within expected bounds?
3. Consistency - are there suspicious jumps or drops?
4. Seasonality - does the pattern match expected seasonal trends?

Provide:
- IS_VALID: [true/false] - overall data quality acceptable?
- CONFIDENCE: [0.0-1.0] - confidence in assessment
- ISSUES: [list any quality issues found]
- RECOMMENDATIONS: [list actionable recommendations]
- SUMMARY: [1-2 sentence summary]"""

        try:
            response = self.client.send_message(
                prompt,
                system=DATA_QUALITY_PROMPT,
                model="haiku",  # Use cheaper model for standard checks
                max_tokens=600
            )

            return self._parse_quality_response(response)

        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return {
                "is_valid": True,  # Don't block on AI errors
                "confidence": 0.0,
                "issues": [],
                "recommendations": [],
                "summary": f"Quality check failed: {e}",
                "error": str(e)
            }

    def check_value(
        self,
        value: float,
        indicator: str,
        date: Optional[str] = None,
        context: Optional[str] = None,
        historical_range: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Check if a specific value seems reasonable.

        Args:
            value: The value to check
            indicator: What the value represents
            date: When the data is from (optional)
            context: Additional context (optional)
            historical_range: Tuple of (min, max) historical values

        Returns:
            Dictionary with:
            - is_reasonable: Whether value seems valid
            - confidence: Confidence level
            - explanation: Why/why not reasonable
            - suggested_action: What to do if suspicious

        Example:
            result = agent.check_value(
                value=-5000,
                indicator="tourist_arrivals",
                context="February 2024"
            )
            # Returns: is_reasonable=False, "Negative arrivals impossible"
        """
        context_text = ""
        if date:
            context_text += f"Date: {date}\n"
        if context:
            context_text += f"Context: {context}\n"
        if historical_range:
            context_text += f"Historical range: {historical_range[0]} to {historical_range[1]}\n"

        prompt = f"""Is this value reasonable for {indicator} in Costa del Sol?

VALUE: {value}
{context_text}
Quickly assess:
1. Is this value physically/logically possible?
2. Is it within expected ranges for this indicator?
3. If suspicious, what might explain it?

Respond concisely:
REASONABLE: [yes/no]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [1 sentence]
ACTION: [keep/verify/flag/reject]"""

        try:
            response = self.client.send_message(
                prompt,
                system=DATA_QUALITY_PROMPT,
                model="haiku",
                max_tokens=200
            )

            return self._parse_value_check(response)

        except Exception as e:
            logger.error(f"Value check failed: {e}")
            return {
                "is_reasonable": True,  # Don't block on errors
                "confidence": 0.0,
                "explanation": f"Check failed: {e}",
                "suggested_action": "verify"
            }

    def explain_issue(
        self,
        issue_description: str,
        data_context: Optional[str] = None
    ) -> str:
        """
        Get a detailed explanation of a data quality issue.

        Args:
            issue_description: Description of the issue
            data_context: Additional context about the data

        Returns:
            Detailed explanation string

        Example:
            explanation = agent.explain_issue(
                "Value dropped 80% month-over-month",
                "Hotel occupancy data for Málaga, March 2020"
            )
        """
        prompt = f"""Explain this data quality issue for Costa del Sol economic data:

ISSUE: {issue_description}
{f'CONTEXT: {data_context}' if data_context else ''}

Provide:
1. Possible causes (list 2-3)
2. How to verify if it's a real issue vs. actual data
3. Recommended action

Be concise but helpful."""

        try:
            response = self.client.send_message(
                prompt,
                system=DATA_QUALITY_PROMPT,
                model="haiku",
                max_tokens=400
            )
            return response.strip()

        except Exception as e:
            return f"Could not explain issue: {e}"

    def suggest_corrections(
        self,
        df: pd.DataFrame,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Suggest corrections for identified data issues.

        Args:
            df: DataFrame with the data
            issues: List of issues (from validate_quality)

        Returns:
            List of suggested corrections with confidence levels
        """
        if not issues:
            return []

        # Prepare issue descriptions
        issue_text = "\n".join([f"- {i}" for i in issues[:5]])

        # Prepare data context
        data_summary = self._prepare_quality_summary(df)

        prompt = f"""Suggest corrections for these data quality issues:

ISSUES:
{issue_text}

DATA CONTEXT:
{data_summary}

For each issue, suggest:
1. Correction approach
2. Confidence level
3. Whether manual review is needed

Be practical and specific."""

        try:
            response = self.client.send_message(
                prompt,
                system=DATA_QUALITY_PROMPT,
                model="sonnet",  # Use better model for corrections
                max_tokens=600
            )

            # Parse suggestions (simplified)
            suggestions = []
            for line in response.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    suggestions.append({
                        "suggestion": line.lstrip("-0123456789. "),
                        "confidence": 0.7,  # Default confidence
                        "needs_review": "manual" in line.lower() or "review" in line.lower()
                    })

            return suggestions

        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return []

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _prepare_quality_summary(self, df: pd.DataFrame) -> str:
        """Prepare data summary focused on quality aspects."""
        summary = []

        summary.append(f"Rows: {len(df)}")
        summary.append(f"Columns: {list(df.columns)}")

        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            summary.append(f"Missing values: {missing[missing > 0].to_dict()}")
        else:
            summary.append("Missing values: None")

        # Value statistics
        if "value" in df.columns:
            values = df["value"].dropna()
            if len(values) > 0:
                summary.append(f"Value range: {values.min():.2f} to {values.max():.2f}")
                summary.append(f"Mean: {values.mean():.2f}, Std: {values.std():.2f}")

                # Check for zeros or negatives
                zeros = (values == 0).sum()
                negatives = (values < 0).sum()
                if zeros > 0:
                    summary.append(f"Zero values: {zeros}")
                if negatives > 0:
                    summary.append(f"Negative values: {negatives}")

        # Date range
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            if len(dates) > 0:
                summary.append(f"Date range: {dates.min()} to {dates.max()}")

        return "\n".join(summary)

    def _parse_quality_response(self, response: str) -> Dict[str, Any]:
        """Parse quality validation response."""
        result = {
            "is_valid": True,
            "confidence": 0.5,
            "issues": [],
            "recommendations": [],
            "summary": ""
        }

        current_section = None

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            upper = line.upper()

            if "IS_VALID:" in upper:
                result["is_valid"] = "true" in line.lower() or "yes" in line.lower()
            elif "CONFIDENCE:" in upper:
                try:
                    conf = float(line.split(":")[-1].strip())
                    result["confidence"] = max(0.0, min(1.0, conf))
                except ValueError:
                    pass
            elif "ISSUES:" in upper:
                current_section = "issues"
            elif "RECOMMENDATION" in upper:
                current_section = "recommendations"
            elif "SUMMARY:" in upper:
                result["summary"] = line.split(":", 1)[-1].strip()
                current_section = None
            elif current_section == "issues" and line.startswith("-"):
                result["issues"].append(line[1:].strip())
            elif current_section == "recommendations" and line.startswith("-"):
                result["recommendations"].append(line[1:].strip())

        return result

    def _parse_value_check(self, response: str) -> Dict[str, Any]:
        """Parse value check response."""
        result = {
            "is_reasonable": True,
            "confidence": 0.5,
            "explanation": "",
            "suggested_action": "keep"
        }

        for line in response.split("\n"):
            line = line.strip()
            upper = line.upper()

            if "REASONABLE:" in upper:
                result["is_reasonable"] = "yes" in line.lower() or "true" in line.lower()
            elif "CONFIDENCE:" in upper:
                try:
                    result["confidence"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "EXPLANATION:" in upper:
                result["explanation"] = line.split(":", 1)[-1].strip()
            elif "ACTION:" in upper:
                action = line.split(":")[-1].strip().lower()
                if action in ["keep", "verify", "flag", "reject"]:
                    result["suggested_action"] = action

        return result


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    print("Testing Data Quality Agent...")

    # Create test data with some issues
    test_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="M"),
        "indicator": "tourist_arrivals",
        "value": [100, 105, 110, 150, 180, 200, -50, 210, 170, None, 105, 100],
        "location": "Málaga"
    })

    print("\nTest Data (with issues):")
    print(test_df)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping live tests.")
    else:
        try:
            agent = DataQualityAgent()

            print("\n1. Validate Quality:")
            result = agent.validate_quality(test_df, "tourist arrivals")
            print(f"   Valid: {result['is_valid']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Issues: {result['issues']}")
            print(f"   Summary: {result['summary']}")

            print("\n2. Check Suspicious Value:")
            check = agent.check_value(
                value=-50,
                indicator="tourist_arrivals",
                date="July 2024",
                context="Peak tourism season"
            )
            print(f"   Reasonable: {check['is_reasonable']}")
            print(f"   Explanation: {check['explanation']}")
            print(f"   Action: {check['suggested_action']}")

        except Exception as e:
            print(f"❌ Error: {e}")
