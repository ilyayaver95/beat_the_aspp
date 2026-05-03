"""
agents/technical_agent.py
==========================
Agent 1: Technical Analysis

Computes all indicators in pure Python/pandas (no LLM needed for the math),
then calls Claude to synthesize them into a score and narrative.

Split of responsibilities:
  Code (market_data.py): EMA/SMA, RSI, ATR, volume analysis, support/resistance,
                         chart patterns, buy/sell zones — all deterministic.
  Claude:                Trend label, score 0-10, summary, short_term_outlook.
"""

from models.report import TechnicalReport, MovingAverages
from tools.market_data import get_market_data, format_candles_for_llm
from cost_tracker import set_context


def run_technical_agent(
    ticker: str,
    period: str = "1y",
    client=None,
) -> tuple[TechnicalReport, dict]:
    """
    Run the Technical Analysis Agent for a given ticker.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        period: Historical data period ("3mo", "6mo", "1y", "2y")
        client: LLM client instance

    Returns:
        (TechnicalReport, market_data_dict)
    """
    if client is None:
        from llm_client import AnthropicLLMClient
        client = AnthropicLLMClient()

    set_context(ticker, "technical_agent")

    # ── Step 1: Compute all indicators ────────────────────────────────
    data = get_market_data(ticker, period)

    # ── Step 2: Build prompts ──────────────────────────────────────────
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(ticker, data)

    # ── Step 3: Call Claude for score + narrative ─────────────────────
    print(f"  [technical_agent] Calling Claude for {ticker} technical analysis...")

    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=TechnicalReport,
    )

    report = response.parsed_output

    # ── Step 4: Override with computed values (more reliable than LLM) ─
    report.ticker = ticker
    report.current_price = data["current_price"]
    report.key_patterns = data["patterns"]

    ma = data["moving_averages"]
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

    rsi = data["rsi"]
    report.rsi_value = rsi["current"]
    report.rsi_condition = rsi["condition"]
    report.rsi_divergence = rsi["divergence"]

    atr = data["atr"]
    report.atr_value = atr["current"]
    report.atr_pct = atr["pct"]

    vol = data["volume_analysis"]
    report.volume_signal = vol["signal"]
    report.volume_description = vol["description"]

    report.support_levels = data["support_levels"]
    report.resistance_levels = data["resistance_levels"]

    bsz = data["buy_sell_zones"]
    report.buy_zone = bsz["buy_zone"]
    report.buy_reasons = bsz["buy_reasons"]
    report.sell_zone = bsz["sell_zone"]
    report.sell_reasons = bsz["sell_reasons"]
    report.action = bsz["action"]
    report.action_reason = bsz["action_reason"]

    print(f"  [technical_agent] Score: {report.score}/10 | {report.trend} ({report.trend_strength})")
    return report, data


def _build_system_prompt() -> str:
    return """You are a professional technical analyst specializing in US equities.
All numerical indicators have already been computed for you by Python code.
Your job is to interpret the data and produce a structured assessment.

SCORING GUIDE (0-10):
  9-10 = Strong technical setup: uptrend, above all MAs, accumulation, pattern near pivot
  7-8  = Good setup: uptrend with minor issues or consolidating in healthy base
  5-6  = Mixed: sideways action, conflicting signals, or weak trend
  3-4  = Weak: below key MAs, or in early downtrend with volume concerns
  0-2  = Bearish: clear downtrend, distribution volume, below all MAs

TREND LABELS:
  trend:          "Bullish" | "Bearish" | "Sideways"
  trend_strength: "Strong" | "Moderate" | "Weak"

STYLE:
  summary:           1-2 sentences. Reference actual price levels and indicator values.
  short_term_outlook: 1 sentence. Specific about what to expect in 1-4 weeks.

Return ONLY the structured JSON — no extra commentary."""


def _build_user_prompt(ticker: str, data: dict) -> str:
    ma = data["moving_averages"]
    rsi = data["rsi"]
    atr = data["atr"]
    vol = data["volume_analysis"]
    trend = data["trend"]
    bsz = data["buy_sell_zones"]
    price = data["current_price"]
    company = data["company_name"]
    patterns = data["patterns"]
    support = data["support_levels"]
    resistance = data["resistance_levels"]
    last = data.get("last_candle", {})

    lines = [
        f"Analyze the technical setup for {company} ({ticker}).",
        f"Current Price: ${price:.2f}",
        "",
        "=== TREND ===",
        f"Direction:  {trend['direction']}",
        f"Strength:   {trend['strength']}",
        f"Above {trend['above_ma_count']}/{trend['total_mas']} moving averages",
        f"Perfectly aligned (bull stack): {trend['perfectly_aligned']}",
        f"Extended above EMA20:           {trend['extended']}",
        f"EMA20 acting as dynamic support: {trend['ema20_dynamic_support']}",
        "",
        "=== MOVING AVERAGES ===",
    ]

    for label, key in [("EMA20", "ma20"), ("EMA50", "ma50"), ("SMA100", "ma100"),
                        ("SMA150", "ma150"), ("SMA200", "ma200")]:
        val = ma.get(key)
        if val:
            pos_key = f"price_vs_{key}"
            pct_key = f"price_pct_from_{key}"
            pos = ma.get(pos_key, "")
            pct = ma.get(pct_key)
            pct_str = f"  ({'+' if pct and pct > 0 else ''}{pct:.1f}%)" if pct is not None else ""
            lines.append(f"  {label}: ${val:.2f}  — price is {pos}{pct_str}")

    lines += [
        f"  Golden Cross: {ma.get('golden_cross', False)}",
        f"  Death Cross:  {ma.get('death_cross', False)}",
        "",
        "=== RSI(14) ===",
        f"  Value:      {rsi['current']:.1f}",
        f"  Condition:  {rsi['condition']}",
        f"  Divergence: {rsi['divergence'] or 'None'}",
        "",
        "=== ATR(14) ===",
        f"  ATR:   ${atr['current']:.2f}  ({atr['pct']:.1f}% of price)",
        "",
        "=== VOLUME ANALYSIS (last 30 days) ===",
        f"  Signal:          {vol['signal']}",
        f"  Description:     {vol['description']}",
        f"  Avg 50-day vol:  {vol['avg_vol_50']:,}",
        f"  High-vol green:  {vol['high_vol_green_days']} days",
        f"  High-vol red:    {vol['high_vol_red_days']} days",
        f"  Volume trend:    {vol['volume_trend']}",
        "",
        "=== CHART PATTERNS ===",
    ]

    if patterns:
        for p in patterns:
            lines.append(f"  • {p}")
    else:
        lines.append("  None detected")

    lines += [
        "",
        "=== SUPPORT & RESISTANCE ===",
        f"  Support levels:    {support}",
        f"  Resistance levels: {resistance}",
        "",
        "=== BUY/SELL ZONES (code-computed) ===",
        f"  Buy zone:    ${bsz['buy_zone']:.2f}" if bsz['buy_zone'] else "  Buy zone: N/A",
        f"  Buy reasons: {bsz['buy_reasons']}",
        f"  Sell zone:   ${bsz['sell_zone']:.2f}" if bsz['sell_zone'] else "  Sell zone: N/A",
        f"  Action:      {bsz['action']} — {bsz['action_reason']}",
        "",
        "=== LAST CANDLE ===",
    ]

    if last:
        lines += [
            f"  Date:   {last.get('date')}",
            f"  OHLC:   O={last.get('open')} H={last.get('high')} "
            f"L={last.get('low')} C={last.get('close')}",
            f"  Bullish: {last.get('is_bullish')}",
            f"  Body %:  {last.get('body_pct')}%  "
            f"Upper wick: {last.get('upper_wick_pct')}%  "
            f"Lower wick: {last.get('lower_wick_pct')}%",
        ]

    lines += [
        "",
        "=== RECENT CANDLES ===",
        format_candles_for_llm(data["df"], n_candles=15),
        "",
        "Based on all the above, provide the TechnicalReport JSON with your "
        "trend assessment, score (0-10), summary, and short_term_outlook.",
    ]

    return "\n".join(lines)
