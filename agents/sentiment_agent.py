"""
agents/sentiment_agent.py
==========================
Agent 3: Sentiment & News Analysis

Fetches recent news via yfinance and sends it to Claude for NLP analysis.
Claude identifies sentiment, key themes, upcoming catalysts, and risks.
"""

from datetime import datetime
from models.report import SentimentReport
from tools.news_scraper import get_news
from cost_tracker import set_context


def run_sentiment_agent(
    ticker: str,
    company_name: str = None,
    client=None,
) -> tuple[SentimentReport, list[dict]]:
    """
    Run the Sentiment Analysis Agent for a given ticker.

    Args:
        ticker:       Stock symbol (e.g., "AAPL")
        company_name: Optional company name for better prompting
        client:       LLM client instance

    Returns:
        (SentimentReport, articles_list)
    """
    if client is None:
        from llm_client import AnthropicLLMClient
        client = AnthropicLLMClient()

    set_context(ticker, "sentiment_agent")

    # ── Step 1: Fetch news ─────────────────────────────────────────────
    articles = get_news(ticker, max_articles=15)

    if not articles:
        print(f"  [sentiment_agent] No news found for {ticker}, using neutral fallback")
        report = SentimentReport(
            ticker=ticker,
            company_name=company_name or ticker,
            articles_analyzed=0,
            date_range="N/A",
            overall_sentiment="Neutral",
            sentiment_score=0.0,
            score=5.0,
            summary="No recent news articles found.",
        )
        return report, []

    # ── Step 2: Build prompts ──────────────────────────────────────────
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(ticker, company_name or ticker, articles)

    # ── Step 3: Call Claude for NLP sentiment analysis ────────────────
    print(f"  [sentiment_agent] Calling Claude for {ticker} sentiment ({len(articles)} articles)...")

    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=SentimentReport,
    )

    report = response.parsed_output

    # ── Step 4: Ensure key fields are populated ────────────────────────
    report.ticker = ticker
    if company_name:
        report.company_name = company_name
    report.articles_analyzed = len(articles)

    # Compute date range from articles
    dates = [a["date"] for a in articles if a["date"] != "Unknown"]
    if dates:
        report.date_range = f"{min(dates)} to {max(dates)}"

    # Populate sources_used
    sources = list({a["publisher"] for a in articles if a["publisher"] != "Unknown"})
    report.sources_used = sources[:5]

    print(f"  [sentiment_agent] Score: {report.score}/10 | {report.overall_sentiment}")
    return report, articles


def _build_system_prompt() -> str:
    return """You are a financial news analyst and market sentiment specialist.
You read news headlines and summaries to assess how the market views a company.

YOUR JOB:
  Analyze the news articles provided and assess the overall sentiment
  toward the company and its stock. Be honest — negative news is negative.

SENTIMENT SCORING:
  overall_sentiment: "Bullish" | "Neutral" | "Bearish"
  sentiment_score:   -1.0 (very bearish) to +1.0 (very bullish)
  score:             0-10 (for the final composite — 5 = neutral)
    9-10 = Overwhelmingly positive news, catalysts ahead, analyst upgrades
    7-8  = Mostly positive, company executing well
    5-6  = Mixed or neutral — no strong signal
    3-4  = More negative than positive, concerns emerging
    0-2  = Significant negative news, earnings miss, downgrades, regulatory issues

WHAT TO IDENTIFY:
  key_themes:          Main topics across the news (e.g., "Contract win", "Guidance raise")
  upcoming_catalysts:  Future events that could move the stock (earnings, product launches)
  risks_mentioned:     Risks or concerns appearing in the news

STYLE:
  summary: 1-2 sentences. Reference specific news items or themes.
  Be specific — mention actual events, not just "positive" or "negative."

Return ONLY the structured JSON."""


def _build_user_prompt(ticker: str, company_name: str, articles: list[dict]) -> str:
    lines = [
        f"Analyze sentiment for {company_name} ({ticker}) from these recent news articles:",
        "",
    ]

    for i, article in enumerate(articles, 1):
        lines.append(f"[{i}] {article['date']} | {article['publisher']}")
        lines.append(f"    TITLE: {article['title']}")
        if article.get("summary"):
            # Truncate long summaries
            summary = article["summary"]
            if len(summary) > 300:
                summary = summary[:300] + "..."
            lines.append(f"    SUMMARY: {summary}")
        lines.append("")

    lines += [
        f"Total articles: {len(articles)}",
        f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "Assess the overall market sentiment for this company based on the above news. "
        "Identify key themes, upcoming catalysts, and risks. "
        "Provide a sentiment_score (-1 to +1) and a composite score (0-10).",
    ]

    return "\n".join(lines)
