# =============================================================================
# News RSS Fetcher
# =============================================================================
# Fetches economic news from Spanish news sources via RSS feeds.
# The news items are used for:
#   - Qualitative context for economic data
#   - Event tracking (infrastructure, policy changes, etc.)
#   - AI-powered summarization
#
# Sources:
#   - Diario Sur (Málaga regional newspaper)
#   - Expansión (Spanish business newspaper)
#   - El Economista
#   - Europa Press (Andalucía section)
# =============================================================================

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd

# feedparser is the standard library for parsing RSS/Atom feeds
try:
    import feedparser
except ImportError:
    feedparser = None
    logging.warning("feedparser not installed. Install with: pip install feedparser")

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# RSS feeds for economic news related to Costa del Sol / Málaga / Andalucía
DEFAULT_FEEDS = [
    {
        "url": "https://www.diariosur.es/rss/2.0/?section=economia",
        "name": "Diario Sur - Economía",
        "category": "regional",
        "language": "es"
    },
    {
        "url": "https://www.expansion.com/rss/andalucia.xml",
        "name": "Expansión - Andalucía",
        "category": "business",
        "language": "es"
    },
    {
        "url": "https://feeds.feedburner.com/eleconomista/economia",
        "name": "El Economista",
        "category": "business",
        "language": "es"
    },
    {
        "url": "https://www.europapress.es/rss/rss.aspx?ch=00285",
        "name": "Europa Press - Andalucía",
        "category": "regional",
        "language": "es"
    },
    {
        "url": "https://www.malagahoy.es/rss/economia/",
        "name": "Málaga Hoy - Economía",
        "category": "regional",
        "language": "es"
    }
]

# Keywords to filter for relevant articles
RELEVANCE_KEYWORDS = [
    # Tourism
    "turismo", "turista", "hotel", "viajero", "visitante",
    "aeropuerto", "crucero", "alojamiento", "ocupación hotelera",
    # Geography
    "málaga", "malaga", "costa del sol", "andalucía", "andalucia",
    "marbella", "torremolinos", "benalmádena", "fuengirola",
    # Economy
    "economía", "economia", "pib", "empleo", "desempleo", "paro",
    "inversión", "inversion", "inmobiliario", "vivienda",
    # Events
    "feria", "congreso", "evento", "festival", "inauguración"
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_feed_date(entry: Dict) -> Optional[datetime]:
    """
    Parse the publication date from an RSS entry.

    RSS feeds use various date formats. feedparser normalizes most of them
    into a time tuple in the 'published_parsed' field.

    Args:
        entry: RSS entry from feedparser

    Returns:
        datetime object or None if parsing fails
    """
    # feedparser provides a pre-parsed date tuple
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6])
        except (TypeError, ValueError):
            pass

    # Try the raw published string
    if hasattr(entry, "published"):
        try:
            return pd.to_datetime(entry.published)
        except Exception:
            pass

    # Try updated date as fallback
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        try:
            return datetime(*entry.updated_parsed[:6])
        except (TypeError, ValueError):
            pass

    return None


def _calculate_relevance_score(title: str, summary: str) -> float:
    """
    Calculate how relevant an article is to our economic monitoring.

    Checks for presence of keywords related to:
    - Costa del Sol / Málaga geography
    - Tourism industry
    - Economic indicators

    Args:
        title: Article title
        summary: Article summary/description

    Returns:
        Score from 0.0 to 1.0 (higher = more relevant)
    """
    text = f"{title} {summary}".lower()

    # Count keyword matches
    matches = sum(1 for keyword in RELEVANCE_KEYWORDS if keyword in text)

    # Calculate score (max out at 5 matches = 1.0)
    score = min(matches / 5.0, 1.0)

    return score


def _clean_html(text: str) -> str:
    """
    Remove HTML tags from text.

    RSS summaries often contain HTML markup that we want to strip out.
    """
    import re

    if not text:
        return ""

    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    clean = clean.replace("&nbsp;", " ")
    clean = clean.replace("&amp;", "&")
    clean = clean.replace("&lt;", "<")
    clean = clean.replace("&gt;", ">")
    clean = clean.replace("&quot;", '"')
    clean = clean.replace("&#39;", "'")

    # Clean up whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean


# =============================================================================
# MAIN FETCHER FUNCTION
# =============================================================================

def fetch_economic_news(
    feeds: Optional[List[Dict]] = None,
    max_articles: int = 50,
    days_back: int = 7,
    min_relevance: float = 0.2
) -> pd.DataFrame:
    """
    Fetch economic news from RSS feeds.

    Retrieves articles from multiple Spanish news sources and filters
    for content relevant to Costa del Sol / Málaga economy.

    Args:
        feeds: List of feed configurations. Each dict should have:
               - url: RSS feed URL
               - name: Source name
               - category: Category (e.g., "regional", "business")
               Default: Uses DEFAULT_FEEDS
        max_articles: Maximum number of articles to return
        days_back: Only include articles from the last N days
        min_relevance: Minimum relevance score (0-1) to include

    Returns:
        DataFrame with columns:
        - date: Publication date
        - title: Article title
        - summary: Article summary (cleaned of HTML)
        - category: Category assigned
        - source: Source name
        - source_url: Article URL
        - relevance_score: Computed relevance (0-1)

    Example:
        news_df = fetch_economic_news(max_articles=20, days_back=3)
        print(news_df[['date', 'title', 'source']])
    """
    if feedparser is None:
        logger.error("feedparser not installed. Install with: pip install feedparser")
        return pd.DataFrame()

    feeds = feeds or DEFAULT_FEEDS
    cutoff_date = datetime.now() - timedelta(days=days_back)

    all_articles = []

    for feed_config in feeds:
        url = feed_config.get("url")
        source_name = feed_config.get("name", "Unknown")
        category = feed_config.get("category", "general")

        logger.info(f"Fetching news from: {source_name}")

        try:
            # Parse the RSS feed
            feed = feedparser.parse(url)

            if feed.bozo:
                # bozo flag indicates a parsing problem
                logger.warning(f"Feed parsing issue for {source_name}: {feed.bozo_exception}")

            for entry in feed.entries:
                # Parse date
                pub_date = _parse_feed_date(entry)
                if not pub_date:
                    continue

                # Filter by date
                if pub_date < cutoff_date:
                    continue

                # Get title and summary
                title = getattr(entry, "title", "")
                summary = _clean_html(getattr(entry, "summary", getattr(entry, "description", "")))
                link = getattr(entry, "link", "")

                # Calculate relevance
                relevance = _calculate_relevance_score(title, summary)

                # Filter by relevance
                if relevance < min_relevance:
                    continue

                all_articles.append({
                    "date": pub_date,
                    "title": title,
                    "summary": summary[:1000] if summary else "",  # Limit length
                    "category": category,
                    "source": source_name,
                    "source_url": link,
                    "relevance_score": relevance
                })

        except Exception as e:
            logger.error(f"Error fetching feed {source_name}: {e}")
            continue

    if not all_articles:
        logger.warning("No relevant articles found in any feed")
        return pd.DataFrame(columns=[
            "date", "title", "summary", "category", "source", "source_url", "relevance_score"
        ])

    # Create DataFrame and sort by date (newest first)
    df = pd.DataFrame(all_articles)
    df = df.sort_values("date", ascending=False)

    # Remove duplicates (same title from different feeds)
    df = df.drop_duplicates(subset=["title"], keep="first")

    # Limit to max articles
    df = df.head(max_articles)

    logger.info(f"Fetched {len(df)} relevant news articles")

    return df


def fetch_single_feed(
    url: str,
    source_name: str = "Custom",
    max_articles: int = 10
) -> pd.DataFrame:
    """
    Fetch news from a single RSS feed.

    Useful for testing or adding custom feeds.

    Args:
        url: RSS feed URL
        source_name: Name for the source
        max_articles: Maximum articles to return

    Returns:
        DataFrame with news articles
    """
    feed_config = [{"url": url, "name": source_name, "category": "custom"}]
    return fetch_economic_news(
        feeds=feed_config,
        max_articles=max_articles,
        min_relevance=0.0  # Don't filter by relevance
    )


# =============================================================================
# KEYWORD EXTRACTION (for categorization)
# =============================================================================

def extract_categories(text: str) -> List[str]:
    """
    Extract economic categories from article text.

    Uses keyword matching to assign categories like:
    - tourism, employment, real_estate, infrastructure, etc.

    Args:
        text: Article title + summary

    Returns:
        List of category strings
    """
    text = text.lower()
    categories = []

    category_keywords = {
        "tourism": ["turismo", "turista", "hotel", "viajero", "ocupación"],
        "employment": ["empleo", "desempleo", "paro", "trabajo", "contratación"],
        "real_estate": ["inmobiliario", "vivienda", "construcción", "hipoteca"],
        "infrastructure": ["aeropuerto", "puerto", "carretera", "ave", "metro"],
        "investment": ["inversión", "inversor", "capital", "financiación"],
        "events": ["feria", "congreso", "evento", "festival", "celebración"]
    }

    for category, keywords in category_keywords.items():
        if any(kw in text for kw in keywords):
            categories.append(category)

    return categories if categories else ["general"]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing News RSS Fetcher...")

    # Test with default feeds
    print("\n1. Fetching from all default feeds:")
    news_df = fetch_economic_news(max_articles=10, days_back=30, min_relevance=0.1)
    print(f"   Articles found: {len(news_df)}")

    if not news_df.empty:
        print("\n   Recent articles:")
        for _, row in news_df.head(5).iterrows():
            print(f"   - [{row['source']}] {row['title'][:60]}...")
            print(f"     Relevance: {row['relevance_score']:.2f}")
            print()

    # Test category extraction
    print("\n2. Testing category extraction:")
    test_texts = [
        "El turismo en Málaga alcanza récord de visitantes",
        "Subida del precio de la vivienda en Costa del Sol",
        "Nuevo hotel de lujo abrirá en Marbella"
    ]
    for text in test_texts:
        cats = extract_categories(text)
        print(f"   '{text[:40]}...' -> {cats}")
