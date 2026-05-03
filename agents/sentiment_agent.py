"""
agents/sentiment_agent.py
=========================
Agent 3: Sentiment Analysis

Scrapes recent news from multiple sources, then uses Claude to perform
Natural Language Processing (NLP) sentiment analysis — identifying:
  - Overall market sentiment (bullish/neutral/bearish)
  - Key themes in the news coverage
  - Upcoming catalysts (earnings, contracts, product launches)
  - Risks mentioned by journalists and analysts
  - Analyst tone (upgrades/downgrades/target changes)

WHY USE CLAUDE FOR SENTIMENT (vs. VADER or FinBERT)?
  Traditional NLP tools like VADER or FinBERT assign sentiment scores
  word-by-word. They can't understand context:
    - "Leonardo DRS wins $2B contract" → VADER might score "wins" as neutral
    - Claude understands this is a MAJOR positive catalyst

  Claude reads the articles like a human analyst — understanding:
    - Implied meaning ("the company revised guidance" = context matters)
    - Relative importance ("the CEO resigned" > "Q2 beat by 1%")
    - Future implications ("EU regulations could hurt margins next year")

CLAUDE API TECHNIQUE — LARGE CONTEXT WINDOW:
  We pass all news articles as a single large context block.
  Claude's 200K context window means we can include 30+ full articles
  without worrying about truncation. This is a major advantage over
  older NLP models that had tiny context windows.
"""

import anthropic
from models.report import SentimentReport
from tools.news_scraper import get_news
from datetime import datetime
from cost_tracker import set_context


def run_sentiment_agent(
    ticker: str,
    company_name: str = None,
    client: anthropic.Anthropic = None,
) -> SentimentReport:
    """
    Run the Sentiment Analysis Agent for a given ticker.

    Args:
        ticker:       Stock symbol (e.g., "DRS")
        company_name: Full company name (fetched from yfinance if not provided)
        client:       Anthropic client instance

    Returns:
        SentimentReport — validated Pydantic model with sentiment scores
    """
    if client is None:
        client = anthropic.Anthropic()

    set_context(ticker, "sentiment_agent")

    # Get company name if not provided
    if not company_name:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        company_name = info.get("longName", ticker)

    # ── Step 1: Scrape news from multiple sources ──────────────────────
    news_data = get_news(ticker, company_name, max_articles=15, days_back=30)

    # ── Step 2: Build the prompt ───────────────────────────────────────
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(ticker, company_name, news_data)

    # ── Step 3: Call Claude with structured output ─────────────────────
    print(f"  [sentiment_agent] Calling Claude for {ticker} sentiment analysis...")
    print(f"  [sentiment_agent] Analyzing {news_data['article_count']} articles...")

    # CLAUDE API TIP — Large Context:
    #   We're potentially sending 30+ articles to Claude.
    #   Claude Opus 4.6 has a 200K token context window, so even
    #   detailed articles won't hit the limit. No chunking needed.
    #   For truly massive datasets (hundreds of articles), you would
    #   want to summarize in batches first, then synthesize.
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=SentimentReport,
    )

    report = response.parsed_output

    # ── Step 4: Fill in metadata ───────────────────────────────────────
    report.ticker = ticker
    report.company_name = company_name
    report.articles_analyzed = news_data["article_count"]
    report.sources_used = news_data["sources_used"]
    report.date_range = f"Last 30 days (as of {datetime.now().strftime('%Y-%m-%d')})"

    if not report.sources_used:
        report.sources_used = news_data["sources_used"]

    print(f"  [sentiment_agent] Score: {report.score}/10 | Sentiment: {report.overall_sentiment}")
    # Return report AND raw articles (needed by HTML report generator for quotes)
    return report, news_data["articles"]


def _build_system_prompt() -> str:
    """
    System prompt giving Claude a financial news analyst persona.

    CLAUDE API TIP — Role Specificity:
      "Analyze sentiment" is vague.
      "You are a sell-side equity research analyst reading news to update
       your investment thesis" is specific. The second version produces
       analysis that actually resembles how real analysts think.
    """
    return """You are a sell-side equity research analyst at a major investment bank.
Your job is to read recent news about a company and assess market sentiment
to inform your investment recommendations.

WHAT YOU'RE LOOKING FOR:
  1. SENTIMENT DRIVERS: What events/news are moving sentiment? (earnings beats,
     contract wins, management changes, regulatory issues, macro headwinds)

  2. CATALYSTS: What upcoming events could materially change the stock price?
     (earnings dates, contract announcements, product launches, regulatory decisions,
     analyst days, M&A rumors)

  3. RISKS: What concerns are being raised by journalists/analysts?
     (customer concentration, margin pressure, competition, geopolitical risk,
     supply chain, litigation, leadership instability)

  4. ANALYST ACTIVITY: Any rating changes, price target updates, or
     initiation of coverage? These are high-signal events.

  5. INSTITUTIONAL ACTIVITY: Any insider buying/selling, hedge fund positions,
     activist investors? (often mentioned in news)

SCORING (sentiment score: -1 to +1):
  +0.7 to +1.0  = Very bullish (strong contract wins, big earnings beat, upgrades)
  +0.3 to +0.6  = Mildly bullish (solid news, minor positive catalysts)
  -0.2 to +0.2  = Neutral (mixed news, no strong direction)
  -0.3 to -0.6  = Mildly bearish (concerns emerging, misses, negative guidance)
  -0.7 to -1.0  = Very bearish (major negative catalyst, scandal, severe miss)

SCORE (0-10):
  Convert sentiment to 0-10 scale: score = (sentiment_score + 1) * 5
  Round to one decimal. Then adjust ±1 for quality/quantity of news.

OUTPUT RULES:
  - Be specific: name actual articles/events, not generic statements
  - Separate facts from speculation (label speculation as "unconfirmed")
  - If no news found: score = 5 (neutral), explain the lack of coverage
  - Return ONLY the structured JSON matching the output schema"""


def _build_user_prompt(ticker: str, company_name: str, news_data: dict) -> str:
    """
    Build the user prompt with all scraped news embedded.
    """
    if news_data["article_count"] == 0:
        news_block = f"No recent news articles found for {company_name} ({ticker}) in the past 30 days."
    else:
        news_block = news_data["formatted_text"]

    return f"""Analyze market sentiment for {company_name} ({ticker}).

{news_block}

Based on these {news_data['article_count']} articles:
1. What is the overall market sentiment? (Bullish / Neutral / Bearish)
2. What are the 3-5 key themes driving coverage?
3. What are the upcoming catalysts to watch?
4. What risks are being mentioned?
5. Provide a sentiment score (-1 to +1) and overall score (0-10).

Write a 2-3 sentence narrative summary of the sentiment picture."""
