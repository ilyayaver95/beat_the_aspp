"""
agents/fundamental_agent.py
============================
Agent 2: Fundamental Analysis

Evaluates a company's financial health using data from yfinance.
Grades each metric against "golden standard" benchmarks and produces
a structured report with an overall fundamental score.

CLAUDE API TECHNIQUE — STRUCTURED OUTPUT WITH PYDANTIC:
  Same as the technical agent — we use client.messages.parse()
  with output_format=FundamentalReport. Claude returns validated JSON.

GRADING SYSTEM (golden standards from CFA/value investing principles):
  Each metric gets a letter grade (A+, A, B, C, D, F) based on
  industry benchmarks. Claude applies these benchmarks contextually,
  considering the company's sector (defense/tech companies have different
  "normal" margins than retail companies, for example).
"""

import anthropic
from models.report import FundamentalReport, FundamentalMetric
from tools.financial_data import get_financial_data


def run_fundamental_agent(
    ticker: str,
    client: anthropic.Anthropic = None,
) -> FundamentalReport:
    """
    Run the Fundamental Analysis Agent for a given ticker.

    Args:
        ticker: Stock symbol (e.g., "DRS")
        client: Anthropic client instance (created if not provided)

    Returns:
        FundamentalReport — validated Pydantic model with graded metrics
    """
    if client is None:
        client = anthropic.Anthropic()

    # ── Step 1: Fetch financial data ───────────────────────────────────
    data = get_financial_data(ticker)

    # ── Step 2: Build the prompt ───────────────────────────────────────
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(ticker, data)

    # ── Step 3: Call Claude with structured output ─────────────────────
    print(f"  [fundamental_agent] Calling Claude for {ticker} fundamental analysis...")

    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=FundamentalReport,
    )

    report = response.parsed_output

    # ── Step 4: Ensure key fields are populated ────────────────────────
    report.ticker = ticker
    report.company_name = data["company_name"]

    info = data["info"]
    report.sector = info.get("sector")
    report.industry = info.get("industry")

    metrics = data["metrics"]
    if metrics.get("market_cap"):
        report.market_cap_billions = round(metrics["market_cap"] / 1e9, 2)
    if metrics.get("revenue_current"):
        report.revenue_ttm_billions = round(metrics["revenue_current"] / 1e9, 2)

    print(f"  [fundamental_agent] Score: {report.score}/10 | {report.company_name}")
    return report


def _build_system_prompt() -> str:
    """
    System prompt giving Claude a CFA analyst persona for fundamental analysis.

    CLAUDE API TIP — Persona Depth:
      The more specific the role, the better the output quality.
      "You are a financial analyst" → generic
      "You are a CFA charterholder with 15 years covering defense stocks" → precise
      The second version produces more nuanced, sector-aware analysis.
    """
    return """You are a CFA charterholder and equity research analyst with 15+ years of experience
analyzing publicly traded companies across sectors including defense, technology, and industrials.

YOUR JOB:
  Evaluate a company's fundamentals using the financial data provided.
  Grade each metric using the golden standard benchmarks listed below.
  Produce an honest, balanced assessment — not a sales pitch.

GRADING BENCHMARKS (use these as guidelines, adjust for sector context):
  Revenue Growth YoY:
    A+ = >25%  |  A = >15%  |  B = >8%  |  C = >0%  |  D = negative  |  F = severe decline

  Net Margin:
    A+ = >25%  |  A = >20%  |  B = >10%  |  C = >5%  |  D = >0%  |  F = negative

  P/E Ratio (trailing):
    A+ = <12  |  A = <18  |  B = <25  |  C = <35  |  D = <50  |  F = >50 or negative

  EPS Growth:
    A+ = >30%  |  A = >20%  |  B = >10%  |  C = >0%  |  D = negative  |  F = severe decline

  ROE (Return on Equity):
    A+ = >30%  |  A = >20%  |  B = >15%  |  C = >10%  |  D = >0%  |  F = negative

  Debt/Equity:
    A+ = <0.2  |  A = <0.5  |  B = <1.0  |  C = <1.5  |  D = <2.5  |  F = >2.5

  Free Cash Flow:
    A+ = strong positive, growing  |  A = positive, stable  |  B = positive, flat
    C = barely positive  |  D = slightly negative  |  F = severely negative

SECTOR CONTEXT:
  Defense companies (DRS, LMT, RTX) often have:
    - Steady but moderate revenue growth (government contracts are multi-year)
    - Moderate margins (5-15% net margin is normal)
    - Conservative balance sheets (government contractors avoid high debt)
    - Strong FCF due to advance payments on contracts
  Adjust your grades accordingly — a 10% net margin is "A" for defense, "C" for software.

SCORING:
  Composite score 0-10:
    9-10 = Exceptional fundamentals, best-in-class
    7-8  = Strong, above average
    5-6  = Mixed, some strengths and weaknesses
    3-4  = Concerning, multiple red flags
    0-2  = Serious fundamental issues

OUTPUT:
  Be specific in comments (e.g., "Revenue grew 12% driven by DoD contracts" not "revenue growth is good").
  List actual key strengths and concerns based on the data.
  Return ONLY the structured JSON matching the output schema."""


def _build_user_prompt(ticker: str, data: dict) -> str:
    """
    Build the user prompt embedding all financial data for Claude to analyze.
    """
    return f"""Analyze the fundamentals for {data['company_name']} ({ticker}).

{data['formatted_text']}

Grade each metric using the benchmarks in your instructions.
Consider the sector context (defense/industrials) when grading.
Provide a composite fundamental score from 0-10 with a clear narrative.

Be specific: reference actual numbers from the data above.
Identify the 2-3 key strengths and 2-3 key concerns."""
