"""
tools/news_scraper.py
=====================
Fetches recent news articles for a ticker using yfinance.

yfinance's ticker.news returns articles from Yahoo Finance including
title, summary, publisher, and publish timestamp.
"""

from datetime import datetime
import yfinance as yf


def get_news(ticker: str, max_articles: int = 15) -> list[dict]:
    """
    Fetch recent news articles for a ticker.

    Args:
        ticker:       Stock ticker symbol (e.g., "AAPL")
        max_articles: Maximum number of articles to return

    Returns:
        List of article dicts with: title, summary, publisher, link, date
    """
    print(f"  [news_scraper] Fetching news for {ticker}...")

    tk = yf.Ticker(ticker)
    raw_news = tk.news or []

    articles = []
    for item in raw_news[:max_articles]:
        # yfinance news format varies by version — handle both
        content = item.get("content", {})

        title = (
            content.get("title")
            or item.get("title")
            or ""
        )
        summary = (
            content.get("summary")
            or item.get("summary")
            or ""
        )
        publisher = (
            content.get("provider", {}).get("displayName")
            or item.get("publisher")
            or "Unknown"
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            or item.get("link")
            or ""
        )
        pub_time = (
            content.get("pubDate")
            or item.get("providerPublishTime")
            or 0
        )

        # Normalize timestamp to readable string
        if isinstance(pub_time, (int, float)) and pub_time > 0:
            date_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d")
        elif isinstance(pub_time, str):
            date_str = pub_time[:10]
        else:
            date_str = "Unknown"

        if title:
            articles.append({
                "title": title,
                "summary": summary,
                "publisher": publisher,
                "link": link,
                "date": date_str,
            })

    print(f"  [news_scraper] Got {len(articles)} articles for {ticker}")
    return articles
