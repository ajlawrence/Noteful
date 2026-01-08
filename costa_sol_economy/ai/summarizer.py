# =============================================================================
# News Summarizer
# =============================================================================
# AI-powered summarization of economic news articles.
#
# Features:
#   - Summarize individual articles
#   - Create digest of multiple articles
#   - Extract key economic events
#   - Classify news sentiment
#   - Identify relevant entities (companies, policies, etc.)
# =============================================================================

import logging
from typing import Optional, Dict, Any, List

import pandas as pd

from .claude_client import ClaudeClient

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

NEWS_SUMMARIZER_PROMPT = """You are a news analyst specializing in Spanish regional economics, particularly tourism and hospitality in the Costa del Sol (Málaga) region.

Your task is to summarize economic news in a clear, factual manner.

When summarizing:
1. Extract the key facts (what happened, when, who is involved)
2. Note any quantitative data (percentages, amounts, dates)
3. Identify the economic impact (positive, negative, or neutral)
4. Highlight relevance to Costa del Sol / Málaga / Andalucía

Keep summaries concise (2-3 sentences per article).
Use professional, neutral language.
If an article is not relevant to regional economics, note that briefly."""


EVENT_EXTRACTOR_PROMPT = """You are analyzing news for economically significant events.

An economically significant event is one that could impact:
- Tourism (arrivals, spending, infrastructure)
- Employment (hiring, layoffs, new businesses)
- Real estate (prices, development, regulations)
- Business climate (investments, regulations, policies)

For each event, extract:
1. Event type (e.g., "infrastructure", "policy", "business")
2. Description (1 sentence)
3. Potential impact (positive/negative/neutral)
4. Affected sectors
5. Timeframe (immediate, short-term, long-term)

Return as structured data."""


# =============================================================================
# NEWS SUMMARIZER CLASS
# =============================================================================

class NewsSummarizer:
    """
    AI agent for summarizing economic news.

    Uses Claude to:
    - Summarize news articles
    - Create news digests
    - Extract economic events
    - Analyze sentiment

    Example usage:
        summarizer = NewsSummarizer()

        # Summarize a single article
        summary = summarizer.summarize_article(title, content)

        # Create a digest from multiple articles
        digest = summarizer.create_digest(news_df)

        # Extract events
        events = summarizer.extract_events(news_df)
    """

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None
    ):
        """
        Initialize the News Summarizer.

        Args:
            claude_client: Existing ClaudeClient instance (or creates new one)
        """
        self.client = claude_client or ClaudeClient()
        logger.info("NewsSummarizer initialized")

    # =========================================================================
    # SINGLE ARTICLE SUMMARIZATION
    # =========================================================================

    def summarize_article(
        self,
        title: str,
        content: str,
        max_length: int = 150
    ) -> Dict[str, Any]:
        """
        Summarize a single news article.

        Args:
            title: Article title
            content: Article text/description
            max_length: Target summary length in words

        Returns:
            Dictionary with:
            - summary: The summary text
            - sentiment: Positive/negative/neutral
            - relevance: How relevant to Costa del Sol (0-1)
            - key_facts: List of key facts extracted

        Example:
            result = summarizer.summarize_article(
                "New hotel opens in Marbella",
                "A luxury 5-star hotel has opened..."
            )
        """
        prompt = f"""Summarize this news article about Spanish/Costa del Sol economy:

TITLE: {title}

CONTENT: {content[:2000]}  # Limit content length

Provide:
1. A concise summary (max {max_length} words)
2. Sentiment (positive/negative/neutral)
3. Relevance to Costa del Sol economy (0.0 to 1.0)
4. 2-3 key facts as bullet points

Format your response as:
SUMMARY: [your summary]
SENTIMENT: [positive/negative/neutral]
RELEVANCE: [0.0-1.0]
KEY FACTS:
- [fact 1]
- [fact 2]
- [fact 3]"""

        try:
            response = self.client.send_message(
                prompt,
                system=NEWS_SUMMARIZER_PROMPT,
                model="haiku",  # Use cheaper model for summarization
                max_tokens=400
            )

            return self._parse_article_summary(response)

        except Exception as e:
            logger.error(f"Article summarization failed: {e}")
            return {
                "summary": f"Summary unavailable: {title}",
                "sentiment": "neutral",
                "relevance": 0.0,
                "key_facts": [],
                "error": str(e)
            }

    def _parse_article_summary(self, response: str) -> Dict[str, Any]:
        """Parse the summary response into structured format."""
        result = {
            "summary": "",
            "sentiment": "neutral",
            "relevance": 0.5,
            "key_facts": []
        }

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("SUMMARY:"):
                result["summary"] = line[8:].strip()
                current_section = "summary"
            elif line.upper().startswith("SENTIMENT:"):
                sentiment = line[10:].strip().lower()
                if sentiment in ["positive", "negative", "neutral"]:
                    result["sentiment"] = sentiment
            elif line.upper().startswith("RELEVANCE:"):
                try:
                    relevance = float(line[10:].strip())
                    result["relevance"] = max(0.0, min(1.0, relevance))
                except ValueError:
                    pass
            elif line.upper().startswith("KEY FACTS:"):
                current_section = "facts"
            elif current_section == "facts" and line.startswith("-"):
                result["key_facts"].append(line[1:].strip())
            elif current_section == "summary" and not line.startswith(("SENTIMENT", "RELEVANCE", "KEY")):
                result["summary"] += " " + line

        result["summary"] = result["summary"].strip()
        return result

    # =========================================================================
    # DIGEST CREATION
    # =========================================================================

    def create_digest(
        self,
        news_df: pd.DataFrame,
        max_articles: int = 10,
        title_column: str = "title",
        summary_column: str = "summary",
        date_column: str = "date"
    ) -> str:
        """
        Create a news digest from multiple articles.

        Args:
            news_df: DataFrame with news articles
            max_articles: Maximum articles to include
            title_column: Column with article titles
            summary_column: Column with article summaries/content
            date_column: Column with publication dates

        Returns:
            Formatted news digest as markdown string

        Example:
            digest = summarizer.create_digest(news_df)
            print(digest)
        """
        if news_df.empty:
            return "# Economic News Digest\n\nNo news articles available."

        # Get most recent articles
        news_df = news_df.copy()
        news_df[date_column] = pd.to_datetime(news_df[date_column])
        recent = news_df.nlargest(max_articles, date_column)

        # Build article list
        articles = []
        for _, row in recent.iterrows():
            title = row.get(title_column, "Untitled")
            content = row.get(summary_column, "")[:500]
            date = row.get(date_column, "")
            if isinstance(date, pd.Timestamp):
                date = date.strftime("%Y-%m-%d")
            articles.append(f"- [{date}] {title}: {content}")

        articles_text = "\n".join(articles)

        prompt = f"""Create a concise news digest for Costa del Sol economic stakeholders.

ARTICLES:
{articles_text}

Create a digest that:
1. Opens with a 1-paragraph overview of the week's economic news
2. Groups related stories by theme (tourism, employment, real estate, etc.)
3. Highlights the most impactful stories
4. Ends with a brief outlook

Format as markdown with headers."""

        try:
            response = self.client.send_message(
                prompt,
                system=NEWS_SUMMARIZER_PROMPT,
                model="sonnet",  # Better model for synthesis
                max_tokens=1000
            )

            return f"# Costa del Sol Economic News Digest\n\n{response}"

        except Exception as e:
            logger.error(f"Digest creation failed: {e}")
            return f"# Economic News Digest\n\nDigest creation failed: {e}"

    # =========================================================================
    # EVENT EXTRACTION
    # =========================================================================

    def extract_events(
        self,
        news_df: pd.DataFrame,
        title_column: str = "title",
        summary_column: str = "summary"
    ) -> List[Dict[str, Any]]:
        """
        Extract economically significant events from news articles.

        Args:
            news_df: DataFrame with news articles
            title_column: Column with titles
            summary_column: Column with content

        Returns:
            List of event dictionaries with:
            - title: Event title
            - event_type: Category of event
            - description: Brief description
            - impact: Potential impact assessment
            - sectors: Affected sectors
            - source_article: Original article title

        Example:
            events = summarizer.extract_events(news_df)
            for event in events:
                print(f"{event['event_type']}: {event['description']}")
        """
        if news_df.empty:
            return []

        # Combine articles for analysis
        articles = []
        for _, row in news_df.head(20).iterrows():
            title = row.get(title_column, "")
            summary = row.get(summary_column, "")[:300]
            if title:
                articles.append(f"• {title}: {summary}")

        articles_text = "\n".join(articles)

        prompt = f"""Extract economically significant events from these news articles:

{articles_text}

For each significant event found, provide:
1. Event type (infrastructure/policy/business/tourism/employment/real_estate)
2. Brief description (1 sentence)
3. Impact (positive/negative/neutral)
4. Affected sectors
5. Which article it came from

Format as a numbered list. Only include genuinely significant events (max 5)."""

        try:
            response = self.client.send_message(
                prompt,
                system=EVENT_EXTRACTOR_PROMPT,
                model="sonnet",
                max_tokens=800
            )

            return self._parse_events(response)

        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return []

    def _parse_events(self, response: str) -> List[Dict[str, Any]]:
        """Parse extracted events from response."""
        events = []
        current_event = {}

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                if current_event:
                    events.append(current_event)
                    current_event = {}
                continue

            # Check for numbered items (new event)
            if line[0].isdigit() and "." in line[:3]:
                if current_event:
                    events.append(current_event)
                current_event = {"description": line.split(".", 1)[-1].strip()}

            # Check for labeled fields
            lower = line.lower()
            if "type:" in lower or "event type:" in lower:
                current_event["event_type"] = line.split(":", 1)[-1].strip()
            elif "impact:" in lower:
                current_event["impact"] = line.split(":", 1)[-1].strip()
            elif "sector" in lower:
                current_event["sectors"] = line.split(":", 1)[-1].strip()
            elif "article:" in lower or "source:" in lower:
                current_event["source_article"] = line.split(":", 1)[-1].strip()

        # Add last event
        if current_event:
            events.append(current_event)

        return events

    # =========================================================================
    # SENTIMENT ANALYSIS
    # =========================================================================

    def analyze_sentiment(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of economic text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with:
            - sentiment: Overall sentiment (positive/negative/neutral)
            - confidence: Confidence level (0-1)
            - reasoning: Brief explanation
        """
        prompt = f"""Analyze the sentiment of this economic news text:

"{text[:1000]}"

Respond with:
SENTIMENT: [positive/negative/neutral]
CONFIDENCE: [0.0-1.0]
REASONING: [1 sentence explanation]"""

        try:
            response = self.client.send_message(
                prompt,
                model="haiku",
                max_tokens=150
            )

            # Parse response
            result = {"sentiment": "neutral", "confidence": 0.5, "reasoning": ""}

            for line in response.split("\n"):
                if "SENTIMENT:" in line.upper():
                    sent = line.split(":")[-1].strip().lower()
                    if sent in ["positive", "negative", "neutral"]:
                        result["sentiment"] = sent
                elif "CONFIDENCE:" in line.upper():
                    try:
                        result["confidence"] = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "REASONING:" in line.upper():
                    result["reasoning"] = line.split(":", 1)[-1].strip()

            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "reasoning": str(e)}


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    print("Testing News Summarizer...")

    # Test data
    test_article = {
        "title": "Málaga airport records 20 million passengers in 2024",
        "content": """Málaga-Costa del Sol Airport has reached a historic milestone,
        recording 20 million passengers in 2024, a 15% increase from the previous year.
        The growth is attributed to new routes from Nordic countries and increased
        British tourism following the post-pandemic recovery. AENA officials
        announced plans to expand Terminal 3 to accommodate the growing demand."""
    }

    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping live tests.")
    else:
        try:
            summarizer = NewsSummarizer()

            print("\n1. Summarize Article:")
            result = summarizer.summarize_article(
                test_article["title"],
                test_article["content"]
            )
            print(f"   Summary: {result['summary']}")
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Relevance: {result['relevance']}")
            print(f"   Key Facts: {result['key_facts']}")

            print("\n2. Sentiment Analysis:")
            sentiment = summarizer.analyze_sentiment(test_article["content"])
            print(f"   Sentiment: {sentiment['sentiment']}")
            print(f"   Confidence: {sentiment['confidence']}")
            print(f"   Reasoning: {sentiment['reasoning']}")

        except Exception as e:
            print(f"❌ Error: {e}")
