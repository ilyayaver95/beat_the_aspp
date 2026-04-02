"""
models/report.py
================
Pydantic data models for all agent outputs and the final synthesized report.

WHY PYDANTIC?
-------------
When we ask Claude to return structured data (scores, lists, metrics),
we need to guarantee the shape of the response. Pydantic models:
  1. Define a strict JSON schema that Claude must follow
  2. Automatically validate Claude's output at runtime
  3. Give us Python objects with dot-access (report.score vs report["score"])
  4. Enable the SDK's `client.messages.parse()` — which handles the
     output_config.format automatically and gives us `response.parsed_output`

CLAUDE API TIP:
  Use `client.messages.parse(output_format=YourModel)` for any situation
  where you need structured, predictable output from Claude. It's more
  reliable than asking Claude to "return JSON" in your prompt.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  AGENT 1 — Technical Analysis Report
# ─────────────────────────────────────────────

class MovingAverages(BaseModel):
    """Current price relative to key moving averages."""
    ma20: Optional[float] = Field(None, description="20-period SMA")
    ma50: Optional[float] = Field(None, description="50-period SMA")
    ma100: Optional[float] = Field(None, description="100-period SMA")
    ma150: Optional[float] = Field(None, description="150-period SMA")
    ma200: Optional[float] = Field(None, description="200-period SMA")
    price_vs_ma20: Optional[str] = Field(None, description="'above' or 'below'")
    price_vs_ma50: Optional[str] = Field(None, description="'above' or 'below'")
    price_vs_ma200: Optional[str] = Field(None, description="'above' or 'below'")
    golden_cross: bool = Field(False, description="MA50 crossed above MA200 recently")
    death_cross: bool = Field(False, description="MA50 crossed below MA200 recently")


class TechnicalReport(BaseModel):
    """
    Structured output from the Technical Analysis Agent.
    Claude fills this in after analyzing candle data and indicators.
    """
    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., description="Latest closing price")
    trend: str = Field(..., description="Overall trend: Bullish / Bearish / Sideways")
    trend_strength: str = Field(..., description="Strength: Strong / Moderate / Weak")

    # Candlestick patterns detected in recent candles
    key_patterns: list[str] = Field(
        default_factory=list,
        description="List of detected candlestick patterns (e.g., 'Hammer on 2024-12-10')"
    )

    moving_averages: MovingAverages = Field(..., description="MA values and price positions")

    # Key price levels
    support_levels: list[float] = Field(
        default_factory=list,
        description="Key support price levels, most recent first"
    )
    resistance_levels: list[float] = Field(
        default_factory=list,
        description="Key resistance price levels, nearest first"
    )

    # Candle anatomy of the most recent candle
    last_candle_body_pct: Optional[float] = Field(
        None, description="Body size as % of full candle range (0-100)"
    )
    last_candle_upper_wick_pct: Optional[float] = Field(
        None, description="Upper wick as % of full candle range"
    )
    last_candle_lower_wick_pct: Optional[float] = Field(
        None, description="Lower wick as % of full candle range"
    )

    score: float = Field(..., ge=0, le=10, description="Technical score 0-10")
    summary: str = Field(..., description="1-2 sentences max, data-driven")
    short_term_outlook: str = Field(..., description="1 sentence: outlook for next 1-4 weeks")


# ─────────────────────────────────────────────
#  AGENT 2 — Fundamental Analysis Report
# ─────────────────────────────────────────────

class FundamentalMetric(BaseModel):
    """A single fundamental metric with its value and grade."""
    value: Optional[float] = None
    grade: str = Field(..., description="A+ / A / B / C / D / F")
    comment: str = Field(..., description="Max 5 words, specific data")


class FundamentalReport(BaseModel):
    """
    Structured output from the Fundamental Analysis Agent.
    All data sourced from yfinance (income statement, balance sheet, cash flow).
    """
    ticker: str
    company_name: str
    sector: Optional[str] = None
    industry: Optional[str] = None

    # Core metrics — each has value + grade + comment
    revenue_growth_yoy: FundamentalMetric = Field(
        ..., description="Year-over-year revenue growth %"
    )
    net_income_margin: FundamentalMetric = Field(
        ..., description="Net income as % of revenue"
    )
    pe_ratio: FundamentalMetric = Field(
        ..., description="Price-to-Earnings ratio"
    )
    eps_growth: FundamentalMetric = Field(
        ..., description="Earnings Per Share growth YoY"
    )
    return_on_equity: FundamentalMetric = Field(
        ..., description="ROE: Net Income / Shareholders' Equity"
    )
    debt_to_equity: FundamentalMetric = Field(
        ..., description="Total Debt / Total Equity"
    )
    free_cash_flow: FundamentalMetric = Field(
        ..., description="Free Cash Flow (operating - capex)"
    )

    # Raw numbers for reference
    market_cap_billions: Optional[float] = None
    revenue_ttm_billions: Optional[float] = None

    score: float = Field(..., ge=0, le=10, description="Fundamental score 0-10")
    summary: str = Field(..., description="1-2 sentences max, cite key numbers")
    key_strengths: list[str] = Field(default_factory=list, description="Max 3, each ≤8 words")
    key_concerns: list[str] = Field(default_factory=list, description="Max 3, each ≤8 words")


# ─────────────────────────────────────────────
#  AGENT 3 — Sentiment Analysis Report
# ─────────────────────────────────────────────

class SentimentReport(BaseModel):
    """
    Structured output from the Sentiment Analysis Agent.
    Combines news scraping with Claude's NLP analysis.
    """
    ticker: str
    company_name: str
    articles_analyzed: int = Field(..., description="Number of news articles processed")
    date_range: str = Field(..., description="Date range of news analyzed")

    # Sentiment breakdown
    overall_sentiment: str = Field(
        ..., description="Bullish / Neutral / Bearish"
    )
    sentiment_score: float = Field(
        ..., ge=-1, le=1,
        description="Sentiment score: -1 (very bearish) to +1 (very bullish)"
    )

    # Themes and catalysts
    key_themes: list[str] = Field(
        default_factory=list,
        description="Main topics from the news (e.g., 'Contract win', 'Earnings beat')"
    )
    upcoming_catalysts: list[str] = Field(
        default_factory=list,
        description="Future events that could move the stock"
    )
    risks_mentioned: list[str] = Field(
        default_factory=list,
        description="Risks or concerns mentioned in news"
    )

    # Source breakdown
    sources_used: list[str] = Field(
        default_factory=list,
        description="News sources that provided articles"
    )

    score: float = Field(..., ge=0, le=10, description="Sentiment score 0-10")
    summary: str = Field(..., description="1-2 sentences max")


# ─────────────────────────────────────────────
#  SYNTHESIZER — Final Analyst Report
# ─────────────────────────────────────────────

class FinalReport(BaseModel):
    """
    The synthesized final report — output of the Orchestrator's Claude call.
    Claude thinks like a Senior Equity Research Analyst when producing this.

    CLAUDE API TIP:
      This is where we use adaptive thinking (thinking: {"type": "adaptive"})
      and effort: "high". The model reasons through all three agent inputs
      before producing a final verdict — similar to how a CFA analyst would
      weigh conflicting signals before publishing a report.
    """
    ticker: str
    company_name: str
    report_date: str = Field(..., description="Date of analysis (YYYY-MM-DD)")

    # Weighted composite scoring
    technical_score: float = Field(..., ge=0, le=10)
    fundamental_score: float = Field(..., ge=0, le=10)
    sentiment_score: float = Field(..., ge=0, le=10)
    composite_score: float = Field(
        ..., ge=0, le=10,
        description="Weighted: 35% technical + 45% fundamental + 20% sentiment"
    )

    # Final verdict
    verdict: str = Field(
        ...,
        description="STRONG BUY / BUY / HOLD / SELL / STRONG SELL"
    )
    confidence_pct: float = Field(
        ..., ge=0, le=100,
        description="Analyst confidence in this verdict (%)"
    )
    time_horizon: str = Field(
        ...,
        description="Recommended holding period (e.g., '3-6 months', '1 year+')"
    )

    # Price context
    current_price: float
    price_target: Optional[str] = Field(
        None,
        description="Estimated price target if determinable (e.g., '$28-32')"
    )

    # Analyst narrative (the most important part)
    analyst_thesis: str = Field(
        ...,
        description="Full analyst thesis: why this verdict, what's the story, key drivers"
    )

    # Risk/opportunity assessment
    key_opportunities: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)

    # Signals that could change the thesis
    bull_case: str = Field(..., description="What needs to happen for the bull case")
    bear_case: str = Field(..., description="What could derail the thesis")

    # What to watch
    watch_for: list[str] = Field(
        default_factory=list,
        description="Key metrics/events to monitor going forward"
    )
