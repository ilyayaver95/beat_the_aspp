"""
tools/news_scraper.py
=====================
Fetches recent news articles about a company from multiple sources.

This is the data source for Agent 3 (Sentiment Analysis).

SOURCES (in priority order, no API key required for defaults):
  1. Yahoo Finance News     — via yfinance ticker.news (most reliable)
  2. Yahoo Finance RSS Feed — via feedparser (structured, fast)
  3. Finviz News            — via requests + BeautifulSoup (scraping)
  4. NewsAPI                — via requests (requires NEWS_API_KEY in .env)

WHY MULTIPLE SOURCES?
  No single source covers everything. A contract announcement might appear
  on Reuters but not Yahoo. Analyst upgrades often hit Finviz first.
  Using 3-5 sources gives the LLM a more complete picture of market
  sentiment, reducing the risk of missing a major catalyst.

USAGE:
  from tools.news_scraper import get_news
  articles = get_news("DRS", "Leonardo DRS Inc", max_articles=30)
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup


def get_news(
    ticker: str,
    company_name: str,
    max_articles: int = 30,
    days_back: int = 30,
) -> dict:
    """
    Fetch recent news from multiple sources and combine them.

    Args:
        ticker:       Stock ticker (e.g., "DRS")
        company_name: Full company name for search context
        max_articles: Maximum total articles to return
        days_back:    How many days of news to include

    Returns:
        Dict with:
          - articles: List of article dicts (title, summary, source, date, url)
          - sources_used: Which sources actually returned data
          - formatted_text: All articles as a text block for the LLM prompt
          - article_count: Total number of articles fetched
    """
    print(f"  [news_scraper] Fetching news for {ticker} ({days_back} days)...")

    all_articles = []
    sources_used = []

    # ── Source 1: Yahoo Finance via yfinance ──────────────────────────
    yf_articles = _get_yfinance_news(ticker)
    if yf_articles:
        all_articles.extend(yf_articles)
        sources_used.append("Yahoo Finance (yfinance)")

    # ── Source 2: Yahoo Finance RSS Feed ─────────────────────────────
    rss_articles = _get_yahoo_rss(ticker)
    if rss_articles:
        all_articles.extend(rss_articles)
        if "Yahoo Finance RSS" not in sources_used:
            sources_used.append("Yahoo Finance RSS")

    # ── Source 3: Finviz News ─────────────────────────────────────────
    finviz_articles = _get_finviz_news(ticker)
    if finviz_articles:
        all_articles.extend(finviz_articles)
        sources_used.append("Finviz")

    # ── Source 4: NewsAPI (optional, requires API key) ────────────────
    news_api_key = os.getenv("NEWS_API_KEY")
    if news_api_key:
        newsapi_articles = _get_newsapi(company_name, news_api_key, days_back)
        if newsapi_articles:
            all_articles.extend(newsapi_articles)
            sources_used.append("NewsAPI")

    # ── Deduplicate by title ──────────────────────────────────────────
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_key = article["title"].lower().strip()[:60]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(article)

    # ── Sort by date (most recent first) ─────────────────────────────
    unique_articles.sort(key=lambda x: x.get("date", ""), reverse=True)

    # ── Limit total articles ──────────────────────────────────────────
    final_articles = unique_articles[:max_articles]

    # ── Format for LLM ────────────────────────────────────────────────
    formatted_text = _format_for_llm(ticker, company_name, final_articles, sources_used)

    print(f"  [news_scraper] Found {len(final_articles)} articles from {len(sources_used)} sources")

    return {
        "articles": final_articles,
        "sources_used": sources_used,
        "article_count": len(final_articles),
        "formatted_text": formatted_text,
    }


def _get_yfinance_news(ticker: str) -> list[dict]:
    """
    Fetch news from Yahoo Finance via yfinance.

    yfinance's ticker.news returns a list of recent articles.
    Each article has: title, link, publisher, providerPublishTime.
    """
    try:
        tk = yf.Ticker(ticker)
        raw_news = tk.news
        if not raw_news:
            return []

        articles = []
        for item in raw_news:
            # Convert Unix timestamp to readable date
            ts = item.get("providerPublishTime", 0)
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else "Unknown"

            articles.append({
                "title": item.get("title", ""),
                "summary": "",  # yfinance doesn't provide summary
                "source": item.get("publisher", "Yahoo Finance"),
                "date": date_str,
                "url": item.get("link", ""),
            })

        return articles

    except Exception as e:
        print(f"  [news_scraper] yfinance news error: {e}")
        return []


def _get_yahoo_rss(ticker: str) -> list[dict]:
    """
    Fetch news from Yahoo Finance RSS feed using feedparser.

    WHY RSS?
      RSS feeds are structured, fast, and don't require authentication.
      feedparser handles all the XML parsing and gives us clean Python dicts.
      This is a great pattern for any site that provides RSS.

    Feed URL format: https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}
    """
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        # feedparser handles the HTTP request + RSS parsing in one call
        feed = feedparser.parse(url)

        if not feed.entries:
            return []

        articles = []
        for entry in feed.entries[:15]:  # Limit to 15 from RSS
            # feedparser normalizes dates via entry.published_parsed
            date_str = "Unknown"
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                date_str = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")

            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],  # Trim long summaries
                "source": entry.get("source", {}).get("title", "Yahoo Finance"),
                "date": date_str,
                "url": entry.get("link", ""),
            })

        return articles

    except Exception as e:
        print(f"  [news_scraper] Yahoo RSS error: {e}")
        return []


def _get_finviz_news(ticker: str) -> list[dict]:
    """
    Scrape recent news headlines from Finviz.com.

    Finviz aggregates news from Bloomberg, Reuters, Seeking Alpha, etc.
    and is a widely-used professional trading tool.

    METHOD: requests + BeautifulSoup HTML parsing
      - requests: fetches the HTML page
      - BeautifulSoup: parses HTML and extracts the news table
      - We target the news table with id="news-table"

    IMPORTANT: Be respectful with scraping — add a delay, set a User-Agent.
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        # Add a small delay to be a good citizen
        time.sleep(1)
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "lxml")
        news_table = soup.find("table", id="news-table")

        if not news_table:
            return []

        articles = []
        current_date = ""

        for row in news_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            # Finviz shows date on first article of each day, then time only
            date_cell = cells[0].text.strip()
            if len(date_cell) > 8:  # Full date + time: "Dec-10-24 07:30AM"
                parts = date_cell.split(" ")
                if parts:
                    try:
                        dt = datetime.strptime(parts[0], "%b-%d-%y")
                        current_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        current_date = datetime.now().strftime("%Y-%m-%d")

            # Extract headline and source
            link_tag = cells[1].find("a")
            if not link_tag:
                continue

            headline = link_tag.text.strip()
            source_span = cells[1].find("span")
            source = source_span.text.strip() if source_span else "Finviz"

            articles.append({
                "title": headline,
                "summary": "",
                "source": f"Finviz via {source}",
                "date": current_date,
                "url": link_tag.get("href", ""),
            })

            if len(articles) >= 20:
                break

        return articles

    except Exception as e:
        print(f"  [news_scraper] Finviz scraping error: {e}")
        return []


def _get_newsapi(company_name: str, api_key: str, days_back: int = 30) -> list[dict]:
    """
    Fetch news from NewsAPI.org (requires free API key).

    NewsAPI searches across hundreds of English-language news sources
    including Reuters, Bloomberg, WSJ, CNBC, etc.

    Free tier: 100 requests/day, 30-day history.
    Sign up: https://newsapi.org/register

    CLAUDE API TIP (parallel with our multi-agent approach):
      NewsAPI is similar to how we structure our agents — instead of
      one big scraper, we use focused, specialized calls to different
      APIs and combine the results. This is the "pipeline" pattern.
    """
    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Search for company name in news articles
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{company_name}"',
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 20,
            "apiKey": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []

        data = response.json()
        articles = []

        for item in data.get("articles", []):
            pub_date = item.get("publishedAt", "")[:10]  # YYYY-MM-DD
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("description", "")[:300],
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "date": pub_date,
                "url": item.get("url", ""),
            })

        return articles

    except Exception as e:
        print(f"  [news_scraper] NewsAPI error: {e}")
        return []


def _format_for_llm(
    ticker: str,
    company_name: str,
    articles: list[dict],
    sources_used: list[str],
) -> str:
    """
    Format all news articles into a structured text block for Claude.

    Each article is presented with its date, source, title, and summary.
    This allows Claude to:
      1. Identify the most recent news first (sentiment can shift fast)
      2. Distinguish between different source types (analyst upgrade vs news)
      3. Read summaries where available for richer context
      4. Notice patterns across multiple articles (recurring themes)

    Returns:
        Text string with all articles, ready for the LLM prompt.
    """
    if not articles:
        return f"No recent news found for {company_name} ({ticker})."

    lines = [
        f"═══ RECENT NEWS: {company_name} ({ticker}) ═══",
        f"Sources: {', '.join(sources_used)}",
        f"Articles: {len(articles)} total",
        "",
    ]

    for i, article in enumerate(articles, 1):
        lines.append(f"[{i}] {article['date']} | {article['source']}")
        lines.append(f"    TITLE: {article['title']}")
        if article.get("summary"):
            lines.append(f"    SUMMARY: {article['summary']}")
        lines.append("")

    return "\n".join(lines)
