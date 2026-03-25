"""
tools/market_data.py
====================
Fetches OHLCV (Open, High, Low, Close, Volume) candlestick data
from Yahoo Finance using the yfinance library.

This is the data source for Agent 1 (Technical Analysis).

WHAT THIS MODULE PROVIDES:
  - Raw candle data (daily & weekly)
  - Pre-computed moving averages (SMA 20/50/100/150/200)
  - Detected candlestick patterns (pure pandas, no TA-Lib needed)
  - Support & Resistance levels via swing high/low detection
  - Candle anatomy for the most recent candle

USAGE:
  from tools.market_data import get_market_data
  data = get_market_data("DRS", period="1y")
"""

import pandas as pd
import numpy as np
from typing import Optional
import yfinance as yf


def get_market_data(ticker: str, period: str = "1y") -> dict:
    """
    Fetch and process all market data needed for technical analysis.

    Args:
        ticker: Stock ticker symbol (e.g., "DRS", "AAPL")
        period: Data period — "6mo", "1y", "2y" (Yahoo Finance format)

    Returns:
        Dictionary with keys:
          - df: Raw OHLCV DataFrame (daily candles)
          - df_weekly: Weekly candle DataFrame
          - moving_averages: Dict of SMA values
          - patterns: List of detected pattern strings
          - support_levels: List of support prices
          - resistance_levels: List of resistance prices
          - last_candle: Dict with anatomy of most recent candle
          - company_name: Full company name
          - current_price: Latest closing price
    """
    print(f"  [market_data] Fetching candles for {ticker}...")

    tk = yf.Ticker(ticker)

    # --- Fetch daily candles ---
    df = tk.history(period=period, interval="1d")

    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")

    # Ensure clean column names
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # --- Fetch weekly candles (for broader trend context) ---
    df_weekly = tk.history(period=period, interval="1wk")
    df_weekly = df_weekly[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # --- Company name ---
    info = tk.info
    company_name = info.get("longName", ticker)
    current_price = float(df["Close"].iloc[-1])

    # --- Compute moving averages ---
    moving_averages = _compute_moving_averages(df)

    # --- Detect candlestick patterns ---
    patterns = _detect_patterns(df)

    # --- Support & Resistance levels ---
    support, resistance = _find_support_resistance(df)

    # --- Most recent candle anatomy ---
    last_candle = _analyze_last_candle(df)

    print(f"  [market_data] Got {len(df)} daily candles for {company_name}")

    return {
        "df": df,
        "df_weekly": df_weekly,
        "company_name": company_name,
        "current_price": current_price,
        "moving_averages": moving_averages,
        "patterns": patterns,
        "support_levels": support,
        "resistance_levels": resistance,
        "last_candle": last_candle,
    }


def _compute_moving_averages(df: pd.DataFrame) -> dict:
    """
    Compute Simple Moving Averages for key periods.

    The classic 'Benjamin Cowen' MA stack: 20, 50, 100, 150, 200.
    - MA20: Short-term momentum
    - MA50: Medium-term trend
    - MA200: Long-term trend (the most watched on Wall Street)
    - Golden Cross: MA50 crosses ABOVE MA200 — bullish signal
    - Death Cross: MA50 crosses BELOW MA200 — bearish signal

    Returns:
        Dict with current MA values, price position, and cross signals.
    """
    close = df["Close"]
    current_price = float(close.iloc[-1])

    ma = {}
    for period in [20, 50, 100, 150, 200]:
        if len(df) >= period:
            ma[f"ma{period}"] = float(close.rolling(period).mean().iloc[-1])
        else:
            ma[f"ma{period}"] = None

    # Price position relative to key MAs
    for period in [20, 50, 200]:
        key = f"ma{period}"
        if ma.get(key):
            ma[f"price_vs_ma{period}"] = "above" if current_price > ma[key] else "below"
            ma[f"price_pct_from_ma{period}"] = round(
                (current_price - ma[key]) / ma[key] * 100, 2
            )

    # Golden Cross / Death Cross detection (look back 10 days)
    ma["golden_cross"] = False
    ma["death_cross"] = False
    if len(df) >= 210:  # Need enough data for both MAs
        ma50_series = close.rolling(50).mean()
        ma200_series = close.rolling(200).mean()
        # Check if there was a cross in the last 20 trading days
        lookback = min(20, len(df) - 200)
        recent_diff = ma50_series.iloc[-lookback:] - ma200_series.iloc[-lookback:]
        if recent_diff.iloc[0] < 0 and recent_diff.iloc[-1] > 0:
            ma["golden_cross"] = True
        elif recent_diff.iloc[0] > 0 and recent_diff.iloc[-1] < 0:
            ma["death_cross"] = True

    return ma


def _detect_patterns(df: pd.DataFrame) -> list[str]:
    """
    Detect classical candlestick patterns using pure pandas math.
    No TA-Lib required — all patterns are computed from OHLC relationships.

    Patterns detected:
      - Doji: Indecision — open ≈ close
      - Hammer: Reversal signal in downtrend (long lower wick)
      - Inverted Hammer: Reversal signal (long upper wick)
      - Shooting Star: Bearish reversal at top (long upper wick)
      - Bullish Engulfing: Strong bullish reversal
      - Bearish Engulfing: Strong bearish reversal
      - Morning Star: 3-candle bullish reversal
      - Evening Star: 3-candle bearish reversal

    Only looks at the most recent 30 candles to surface relevant signals.

    Returns:
        List of strings describing each detected pattern with its date.
    """
    # Work on a copy of recent candles
    recent = df.tail(30).copy()

    # ── Candle anatomy metrics ──────────────────────────────────────
    recent["body"] = (recent["Close"] - recent["Open"]).abs()
    recent["candle_range"] = recent["High"] - recent["Low"]
    recent["upper_wick"] = recent["High"] - recent[["Open", "Close"]].max(axis=1)
    recent["lower_wick"] = recent[["Open", "Close"]].min(axis=1) - recent["Low"]
    recent["is_bullish"] = recent["Close"] > recent["Open"]

    # Avoid division by zero
    range_safe = recent["candle_range"].replace(0, np.nan)
    recent["body_pct"] = recent["body"] / range_safe  # 0–1

    patterns = []

    for i in range(1, len(recent)):
        row = recent.iloc[i]
        prev = recent.iloc[i - 1]
        date_str = recent.index[i].strftime("%Y-%m-%d")
        rng = row["candle_range"]

        if rng == 0:
            continue  # Skip holiday/no-trade candles

        # ── DOJI ────────────────────────────────────────────────────
        # Body is tiny relative to range — market is undecided
        if row["body_pct"] < 0.10:
            patterns.append(f"Doji on {date_str} (indecision — watch for breakout)")

        # ── HAMMER ──────────────────────────────────────────────────
        # Long lower wick, small body near top — bullish reversal signal
        elif (
            row["lower_wick"] > 2 * row["body"]
            and row["upper_wick"] < 0.3 * rng
            and row["is_bullish"]
        ):
            patterns.append(f"Hammer on {date_str} (bullish reversal signal)")

        # ── INVERTED HAMMER / SHOOTING STAR ─────────────────────────
        # Long upper wick, small body — context determines meaning
        elif (
            row["upper_wick"] > 2 * row["body"]
            and row["lower_wick"] < 0.3 * rng
        ):
            if row["is_bullish"]:
                patterns.append(f"Inverted Hammer on {date_str} (potential bullish reversal)")
            else:
                patterns.append(f"Shooting Star on {date_str} (bearish reversal warning)")

        # ── BULLISH ENGULFING ────────────────────────────────────────
        # Bullish candle fully engulfs the prior bearish candle
        elif (
            row["is_bullish"]
            and not prev["is_bullish"]
            and row["Open"] <= prev["Close"]
            and row["Close"] >= prev["Open"]
        ):
            patterns.append(f"Bullish Engulfing on {date_str} (strong bullish reversal)")

        # ── BEARISH ENGULFING ────────────────────────────────────────
        # Bearish candle fully engulfs the prior bullish candle
        elif (
            not row["is_bullish"]
            and prev["is_bullish"]
            and row["Open"] >= prev["Close"]
            and row["Close"] <= prev["Open"]
        ):
            patterns.append(f"Bearish Engulfing on {date_str} (strong bearish reversal)")

    # ── MORNING STAR (3-candle) ──────────────────────────────────────
    # Bearish candle → small body → Bullish candle = bottom reversal
    for i in range(2, len(recent)):
        c1, c2, c3 = recent.iloc[i - 2], recent.iloc[i - 1], recent.iloc[i]
        date_str = recent.index[i].strftime("%Y-%m-%d")
        if (
            not c1["is_bullish"]               # C1: bearish
            and c2["body_pct"] < 0.3           # C2: small body (indecision)
            and c3["is_bullish"]               # C3: bullish
            and c3["Close"] > (c1["Open"] + c1["Close"]) / 2  # Closes above C1 midpoint
        ):
            patterns.append(f"Morning Star around {date_str} (3-candle bullish reversal)")

    # ── EVENING STAR (3-candle) ──────────────────────────────────────
    # Bullish candle → small body → Bearish candle = top reversal
    for i in range(2, len(recent)):
        c1, c2, c3 = recent.iloc[i - 2], recent.iloc[i - 1], recent.iloc[i]
        date_str = recent.index[i].strftime("%Y-%m-%d")
        if (
            c1["is_bullish"]                   # C1: bullish
            and c2["body_pct"] < 0.3           # C2: small body
            and not c3["is_bullish"]           # C3: bearish
            and c3["Close"] < (c1["Open"] + c1["Close"]) / 2  # Closes below C1 midpoint
        ):
            patterns.append(f"Evening Star around {date_str} (3-candle bearish reversal)")

    return patterns


def _find_support_resistance(df: pd.DataFrame, window: int = 10) -> tuple[list, list]:
    """
    Find support and resistance levels using swing high/low detection.

    METHOD:
      A swing LOW (support) is a candle whose Low is the lowest
      in the surrounding `window` candles on both sides.
      A swing HIGH (resistance) is the opposite.

    We then cluster nearby levels to avoid duplicates and return
    the most significant ones (sorted by recency).

    Args:
        df: OHLCV DataFrame
        window: How many candles on each side to check

    Returns:
        (support_levels, resistance_levels) — lists of float prices
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_lows = []
    swing_highs = []

    for i in range(window, n - window):
        # Swing low: local minimum
        if lows[i] == min(lows[i - window : i + window + 1]):
            swing_lows.append(lows[i])
        # Swing high: local maximum
        if highs[i] == max(highs[i - window : i + window + 1]):
            swing_highs.append(highs[i])

    # Cluster nearby levels (within 1% of each other)
    def cluster_levels(levels: list, threshold_pct: float = 0.01) -> list:
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = [levels[0]]
        for level in levels[1:]:
            if abs(level - clustered[-1]) / clustered[-1] > threshold_pct:
                clustered.append(level)
        return clustered

    support = cluster_levels(swing_lows)
    resistance = cluster_levels(swing_highs)

    # Return most recent/relevant (last 5 of each)
    current_price = float(df["Close"].iloc[-1])
    support = sorted(
        [s for s in support if s < current_price],
        reverse=True
    )[:5]
    resistance = sorted(
        [r for r in resistance if r > current_price]
    )[:5]

    return [round(s, 2) for s in support], [round(r, 2) for r in resistance]


def _analyze_last_candle(df: pd.DataFrame) -> dict:
    """
    Analyze the anatomy of the most recent candle.

    Returns breakdown of body, upper wick, and lower wick
    as percentages of the total candle range — useful for
    understanding buying vs selling pressure.

    Example:
      body_pct=70%, upper_wick_pct=5%, lower_wick_pct=25%
      → Strong bullish candle with buying pressure, mild selling at top.
    """
    last = df.iloc[-1]
    rng = last["High"] - last["Low"]

    if rng == 0:
        return {"body_pct": 0, "upper_wick_pct": 0, "lower_wick_pct": 0}

    body = abs(last["Close"] - last["Open"])
    upper_wick = last["High"] - max(last["Open"], last["Close"])
    lower_wick = min(last["Open"], last["Close"]) - last["Low"]

    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "open": round(float(last["Open"]), 2),
        "high": round(float(last["High"]), 2),
        "low": round(float(last["Low"]), 2),
        "close": round(float(last["Close"]), 2),
        "volume": int(last["Volume"]),
        "is_bullish": last["Close"] > last["Open"],
        "body_pct": round(body / rng * 100, 1),
        "upper_wick_pct": round(upper_wick / rng * 100, 1),
        "lower_wick_pct": round(lower_wick / rng * 100, 1),
    }


def format_candles_for_llm(df: pd.DataFrame, n_candles: int = 30) -> str:
    """
    Format recent candle data as a compact string for the LLM prompt.

    WHY THIS MATTERS:
      LLMs can't "see" a chart, but they CAN read a well-formatted
      table of numbers and reason about patterns from it.
      This is how we bridge the gap between raw market data and LLM analysis.

    Args:
        df: OHLCV DataFrame
        n_candles: Number of recent candles to include

    Returns:
        Multi-line string with date, OHLCV per row.
    """
    recent = df.tail(n_candles).copy()
    lines = ["Date        | Open    | High    | Low     | Close   | Volume"]
    lines.append("-" * 70)
    for date, row in recent.iterrows():
        lines.append(
            f"{date.strftime('%Y-%m-%d')} | "
            f"{row['Open']:>7.2f} | "
            f"{row['High']:>7.2f} | "
            f"{row['Low']:>7.2f} | "
            f"{row['Close']:>7.2f} | "
            f"{int(row['Volume']):>10,}"
        )
    return "\n".join(lines)
