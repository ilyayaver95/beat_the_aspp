"""
agents/technical_agent.py
=========================
Agent 1: Technical Analysis — Micha Stocks style

HOW IT WORKS:
  1. Fetch OHLCV data via tools/market_data.py (yfinance)
  2. All indicators computed in code: EMA/SMA, RSI, ATR, Volume, Patterns, Buy/Sell zones
  3. Send a COMPACT summary to Claude (not raw candles) — saves tokens
  4. Claude only adds: score, narrative summary, and short-term outlook
  5. All other fields are filled from code computations

TOKEN OPTIMIZATION:
  - Indicators, buy/sell zones, patterns all computed in Python
  - Claude gets a pre-digested summary, not 30 rows of candle data
  - Claude's job: score 0-10, write 2 sentences, 1-sentence outlook
"""

import anthropic
from pydantic import BaseModel, Field
from models.report import TechnicalReport, MovingAverages
from tools.market_data import get_market_data
from cost_tracker import set_context


class _ClaudeOutput(BaseModel):
    """Minimal model for what Claude returns — keeps grammar compilation fast."""
    score: float = Field(..., ge=0, le=10, description="Technical score 0-10")
    trend: str = Field(..., description="Bullish / Bearish / Sideways")
    trend_strength: str = Field(..., description="Strong / Moderate / Weak")
    summary: str = Field(..., description="1-2 sentences, reference price levels")
    short_term_outlook: str = Field(..., description="1 sentence outlook")


def run_technical_agent(
    ticker: str,
    period: str = "1y",
    client: anthropic.Anthropic = None,
) -> TechnicalReport:
    """
    Run the Technical Analysis Agent for a given ticker.

    Returns:
        (TechnicalReport, market_data_dict)
    """
    if client is None:
        client = anthropic.Anthropic()

    set_context(ticker, "technical_agent")

    # ── Step 1: Fetch market data (all indicators computed here) ──────
    data = get_market_data(ticker, period=period)

    # ── Step 2: Build compact prompt (no raw candle table) ────────────
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(ticker, data)

    # ── Step 3: Call Claude for score + narrative only ─────────────────
    print(f"  [technical_agent] Calling Claude for {ticker} technical analysis...")

    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=_ClaudeOutput,
    )

    claude_out = response.parsed_output

    # ── Step 4: Build full report from code + Claude's score/narrative ─
    report = TechnicalReport(
        ticker=ticker,
        current_price=data["current_price"],
        trend=claude_out.trend,
        trend_strength=claude_out.trend_strength,
        moving_averages=MovingAverages(),
        score=claude_out.score,
        summary=claude_out.summary,
        short_term_outlook=claude_out.short_term_outlook,
    )
    report = _fill_computed_fields(report, data, ticker)

    print(f"  [technical_agent] Score: {report.score}/10 | Trend: {report.trend}")
    return report, data


def _fill_computed_fields(report: TechnicalReport, data: dict, ticker: str) -> TechnicalReport:
    """Fill in all fields that are computed in code, not by Claude."""
    ma = data["moving_averages"]

    report.ticker = ticker
    report.current_price = data["current_price"]

    # Moving averages
    report.moving_averages = MovingAverages(
        ma20=ma.get("ma20"),
        ma50=ma.get("ma50"),
        ma100=ma.get("ma100"),
        ma150=ma.get("ma150"),
        ma200=ma.get("ma200"),
        price_vs_ma20=ma.get("price_vs_ma20"),
        price_vs_ma50=ma.get("price_vs_ma50"),
        price_vs_ma150=ma.get("price_vs_ma150"),
        price_vs_ma200=ma.get("price_vs_ma200"),
        golden_cross=ma.get("golden_cross", False),
        death_cross=ma.get("death_cross", False),
    )

    # RSI
    rsi = data["rsi"]
    report.rsi_value = round(rsi["current"], 1)
    report.rsi_condition = rsi["condition"]
    report.rsi_divergence = rsi["divergence"]

    # ATR
    atr = data["atr"]
    report.atr_value = atr["current"]
    report.atr_pct = atr["pct"]

    # Volume
    vol = data["volume_analysis"]
    report.volume_signal = vol["signal"]
    report.volume_description = vol["description"]

    # Support / Resistance
    report.support_levels = data["support_levels"]
    report.resistance_levels = data["resistance_levels"]

    # Patterns
    report.key_patterns = data["patterns"]

    # Buy/Sell zones
    bs = data["buy_sell_zones"]
    report.buy_zone = bs["buy_zone"]
    report.buy_reasons = bs["buy_reasons"]
    report.sell_zone = bs["sell_zone"]
    report.sell_reasons = bs["sell_reasons"]
    report.action = bs["action"]
    report.action_reason = bs["action_reason"]

    return report


def _build_system_prompt() -> str:
    """Micha Stocks style analyst — concise, data-driven."""
    return """You are a technical analyst in the style of Micha Stocks Academy.
You receive pre-computed indicators and your job is to:
1. Score the stock 0-10 (0-3 bearish, 4-6 neutral, 7-10 bullish)
2. Write a 1-2 sentence summary referencing specific price levels
3. Write a 1-sentence short-term outlook

STYLE:
  - Think in probabilities, not certainties
  - Reference specific price levels (e.g., "the $24 area held as support")
  - Focus on: trend, volume confirmation, RSI, key levels
  - Keep it actionable and concise

CONSTRAINTS:
  - Only use data provided — do not make up numbers
  - summary: 1-2 sentences max, mention specific price levels
  - short_term_outlook: 1 sentence only
  - Return ONLY the structured JSON"""


def _build_user_prompt(ticker: str, data: dict) -> str:
    """
    Build a COMPACT prompt with pre-digested indicators.
    No raw candle table — everything is already analyzed.
    """
    ma = data["moving_averages"]
    rsi = data["rsi"]
    atr = data["atr"]
    vol = data["volume_analysis"]
    trend = data["trend"]
    bs = data["buy_sell_zones"]
    patterns = data["patterns"]
    gaps = data["gaps"]
    current_price = data["current_price"]
    company_name = data["company_name"]

    # MA summary lines
    ma_lines = []
    for label, key in [("EMA20", "ma20"), ("EMA50", "ma50"), ("SMA150", "ma150"), ("SMA200", "ma200")]:
        val = ma.get(key)
        pos = ma.get(f"price_vs_{key}")
        pct = ma.get(f"price_pct_from_{key}")
        if val:
            ma_lines.append(f"  {label}: ${val:.2f} — price {pos} ({pct:+.1f}%)")

    cross = ""
    if ma.get("golden_cross"):
        cross = "  GOLDEN CROSS (EMA50 above SMA200) — bullish"
    elif ma.get("death_cross"):
        cross = "  DEATH CROSS (EMA50 below SMA200) — bearish"

    prompt = f"""Analyze {company_name} ({ticker}) at ${current_price:.2f}

TREND: {trend['direction']} ({trend['strength']})
  MAs aligned: {"Yes" if trend['perfectly_aligned'] else "No"}
  Extended from EMA20: {"Yes — caution" if trend['extended'] else "No"}
  EMA20 dynamic support: {"Yes" if trend['ema20_dynamic_support'] else "No"}

MOVING AVERAGES:
{chr(10).join(ma_lines)}
{cross}

RSI(14): {rsi['current']:.1f} — {rsi['condition']}
{f"  Divergence: {rsi['divergence']}" if rsi['divergence'] else "  No divergence"}

ATR(14): ${atr['current']:.2f} ({atr['pct']:.1f}% of price)

VOLUME: {vol['signal']} — {vol['description']}

SUPPORT: {', '.join(f'${s:.2f}' for s in data['support_levels']) or 'None found'}
RESISTANCE: {', '.join(f'${r:.2f}' for r in data['resistance_levels']) or 'None found'}

PATTERNS: {'; '.join(patterns) if patterns else 'None detected'}
GAPS: {'; '.join(f"{g['type']} {g['pct']}% on {g['date']}" for g in gaps) if gaps else 'None'}

ACTION: {bs['action']} — {bs['action_reason']}
BUY ZONE: {f"${bs['buy_zone']:.2f}" if bs['buy_zone'] else 'N/A'}
SELL ZONE: {f"${bs['sell_zone']:.2f}" if bs['sell_zone'] else 'N/A'}

Score 0-10 and provide summary + outlook."""

    return prompt
