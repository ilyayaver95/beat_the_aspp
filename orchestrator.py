"""
orchestrator.py
===============
The Orchestrator: coordinates all three agents and synthesizes their
reports into a final analyst verdict, then generates an HTML report.

HOW IT WORKS:
  1. Run all 3 agents IN PARALLEL using ThreadPoolExecutor
  2. Collect structured reports + raw data from each agent
  3. Call Claude Opus 4.6 with Adaptive Thinking for synthesis
  4. Generate an HTML report with candlestick chart, tables, quotes

CLAUDE API TECHNIQUES:
  - Adaptive Thinking: deep reasoning for the synthesis step
  - Effort "high": maximum quality for the final verdict
  - Streaming: real-time output to the user during synthesis
  - Parallel execution: 3 agents run simultaneously (~3x faster)
"""

import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os

from models.report import TechnicalReport, FundamentalReport, SentimentReport, FinalReport
from agents.technical_agent import run_technical_agent
from agents.fundamental_agent import run_fundamental_agent
from agents.sentiment_agent import run_sentiment_agent


def run_analysis(
    ticker: str,
    period: str = "1y",
    stream_output: bool = True,
) -> FinalReport:
    """
    Main orchestration function. Runs all 3 agents in parallel,
    synthesizes results, and generates an HTML report.

    Args:
        ticker:        Stock ticker to analyze (e.g., "DRS")
        period:        Historical price period for technical analysis
        stream_output: If True, streams the synthesis to stdout in real-time

    Returns:
        FinalReport — the complete analyst verdict
    """
    client = anthropic.Anthropic()

    print(f"\n{'='*60}")
    print(f"  ANALYZING: {ticker.upper()}")
    print(f"{'='*60}")
    print(f"  Running 3 agents in parallel...\n")

    # ── STEP 1: Run all 3 agents in parallel ──────────────────────────
    tech_report = fund_report = sent_report = None
    market_data = news_articles = None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_technical_agent, ticker, period, client): "technical",
            executor.submit(run_fundamental_agent, ticker, client): "fundamental",
            executor.submit(run_sentiment_agent, ticker, None, client): "sentiment",
        }

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                if agent_name == "technical":
                    tech_report, market_data = result   # (TechnicalReport, raw_data)
                elif agent_name == "fundamental":
                    fund_report = result                 # FundamentalReport
                elif agent_name == "sentiment":
                    sent_report, news_articles = result  # (SentimentReport, articles)
                print(f"  ✓ {agent_name.capitalize()} agent complete")
            except Exception as e:
                print(f"  ✗ {agent_name.capitalize()} agent failed: {e}")
                import traceback; traceback.print_exc()

    # Fallbacks for failed agents
    if not tech_report:
        tech_report = _fallback_technical(ticker)
    if not fund_report:
        fund_report = _fallback_fundamental(ticker)
    if not sent_report:
        sent_report = _fallback_sentiment(ticker)
    if news_articles is None:
        news_articles = []

    # ── STEP 2: Synthesize into final report ──────────────────────────
    print(f"\n{'─'*60}")
    print(f"  SYNTHESIZING ANALYST REPORT...")
    print(f"{'─'*60}\n")

    final_report = _synthesize(
        ticker=ticker,
        tech=tech_report,
        fund=fund_report,
        sent=sent_report,
        client=client,
        stream=stream_output,
    )

    # ── STEP 3: Generate HTML report ──────────────────────────────────
    from report_generator import generate_html_report
    html_path = generate_html_report(
        ticker=ticker,
        tech_report=tech_report,
        fund_report=fund_report,
        sent_report=sent_report,
        final_report=final_report,
        market_data=market_data,
        news_articles=news_articles,
    )
    if html_path:
        print(f"\n  HTML report saved: {html_path}")

    return final_report


def _synthesize(
    ticker: str,
    tech: TechnicalReport,
    fund: FundamentalReport,
    sent: SentimentReport,
    client: anthropic.Anthropic,
    stream: bool = True,
) -> FinalReport:
    """
    Call Claude Opus 4.6 with Adaptive Thinking to synthesize all
    three agent reports into a final analyst verdict.
    """
    system_prompt = _build_synthesis_system_prompt()
    user_prompt = _build_synthesis_user_prompt(ticker, tech, fund, sent)

    composite = round(tech.score * 0.35 + fund.score * 0.45 + sent.score * 0.20, 1)

    if stream:
        print(f"  Composite Score (pre-synthesis): {composite}/10")
        print(f"  Tech: {tech.score} x 35% | Fund: {fund.score} x 45% | Sent: {sent.score} x 20%")
        print(f"\n{'='*60}")
        print(f"  ANALYST REPORT — {ticker.upper()}")
        print(f"{'='*60}\n")

        full_text = ""
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream_obj:
            for event in stream_obj:
                if (
                    hasattr(event, "type")
                    and event.type == "content_block_delta"
                    and hasattr(event, "delta")
                    and hasattr(event.delta, "type")
                    and event.delta.type == "text_delta"
                ):
                    chunk = event.delta.text
                    print(chunk, end="", flush=True)
                    full_text += chunk
        print("\n")

        final_report = _parse_streamed_report(full_text, ticker, tech, fund, sent, composite, client)

    else:
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=FinalReport,
        )
        final_report = response.parsed_output

    final_report.ticker = ticker
    final_report.technical_score = tech.score
    final_report.fundamental_score = fund.score
    final_report.sentiment_score = sent.score
    final_report.composite_score = composite
    final_report.current_price = tech.current_price
    final_report.report_date = datetime.now().strftime("%Y-%m-%d")
    return final_report


def _build_synthesis_system_prompt() -> str:
    return """You are a Senior Equity Research Analyst and Portfolio Strategist at a top-tier
investment bank with 20+ years of experience. You synthesize technical, fundamental,
and sentiment signals into clear, actionable investment recommendations.

WEIGHTING FRAMEWORK:
  Fundamental Analysis: 45% (most important for medium/long-term)
  Technical Analysis:   35% (timing and entry points)
  Sentiment Analysis:   20% (confirming or contradicting signal)

VERDICT SCALE:
  8.5-10.0 → STRONG BUY   | 7.0-8.4 → BUY  | 5.0-6.9 → HOLD
  3.0-4.9  → SELL          | 0.0-2.9 → STRONG SELL

WHEN SIGNALS CONFLICT:
  - Strong fundamentals + weak technicals → BUY with timing caution
  - Weak fundamentals + strong technicals → HOLD/SELL, don't chase
  - Strong everything + poor sentiment → BUY (news lags reality)
  - Good sentiment + poor fundamentals → Cautious

STYLE: Write like a real analyst report. Confident, specific, data-driven.
Cite actual numbers. Reference real price levels. Be decisive.
The analyst_thesis must be 3-4 full sentences: the story, opportunity, and risk.

Return a JSON object matching the FinalReport schema exactly."""


def _build_synthesis_user_prompt(ticker, tech, fund, sent) -> str:
    composite = round(tech.score * 0.35 + fund.score * 0.45 + sent.score * 0.20, 1)

    def fmt(r):
        return json.dumps(r.model_dump(), indent=2, default=str)

    return f"""Synthesize these three reports for {fund.company_name} ({ticker}).

COMPOSITE: {composite}/10  (Tech {tech.score}x0.35 + Fund {fund.score}x0.45 + Sent {sent.score}x0.20)

=== TECHNICAL REPORT ===
{fmt(tech)}

=== FUNDAMENTAL REPORT ===
{fmt(fund)}

=== SENTIMENT REPORT ===
{fmt(sent)}

Produce the complete FinalReport JSON with your definitive verdict.
Think through where signals agree, where they conflict, and what matters most."""


def _parse_streamed_report(text, ticker, tech, fund, sent, composite, client) -> FinalReport:
    """Convert the streamed analyst narrative into a structured FinalReport."""
    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=4096,
        system="""Extract a structured FinalReport JSON from the analyst narrative below.
Derive all fields from the text. Return ONLY valid JSON matching the FinalReport schema.""",
        messages=[{"role": "user", "content": f"""Analyst narrative for {ticker}:

{text}

Context: ticker={ticker}, company={fund.company_name},
tech_score={tech.score}, fund_score={fund.score}, sent_score={sent.score},
composite={composite}, price={tech.current_price},
date={datetime.now().strftime('%Y-%m-%d')}"""}],
        output_format=FinalReport,
    )
    return response.parsed_output


# ── Fallbacks ─────────────────────────────────────────────────────────

def _fallback_technical(ticker):
    from models.report import MovingAverages
    return TechnicalReport(
        ticker=ticker, current_price=0.0, trend="Unknown", trend_strength="Unknown",
        moving_averages=MovingAverages(), score=5.0,
        summary="Technical data unavailable.", short_term_outlook="Unknown.",
    )

def _fallback_fundamental(ticker):
    from models.report import FundamentalMetric
    n = FundamentalMetric(grade="C", comment="Data unavailable")
    return FundamentalReport(
        ticker=ticker, company_name=ticker,
        revenue_growth_yoy=n, net_income_margin=n, pe_ratio=n,
        eps_growth=n, return_on_equity=n, debt_to_equity=n, free_cash_flow=n,
        score=5.0, summary="Fundamental data unavailable.",
    )

def _fallback_sentiment(ticker):
    return SentimentReport(
        ticker=ticker, company_name=ticker, articles_analyzed=0,
        date_range="Unknown", overall_sentiment="Neutral", sentiment_score=0.0,
        score=5.0, summary="Sentiment data unavailable.",
    )
