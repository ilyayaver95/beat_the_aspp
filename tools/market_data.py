"""
tools/market_data.py
====================
Fetches OHLCV data and computes all technical indicators in pure Python/pandas.

Micha Stocks style analysis — all decisions are hardcoded, no LLM needed for:
  - EMA/SMA (20, 50, 150, 200) with dynamic support detection
  - RSI (14) with overbought/oversold and divergence detection
  - ATR (14) for volatility measurement
  - Volume analysis: institutional buying vs distribution
  - Chart patterns: Cup & Handle, VCP, Flat Base
  - Gap detection (up/down gaps)
  - Support & Resistance as "areas of interest"
  - Buy/Sell zone computation
"""

import pandas as pd
import numpy as np
from typing import Optional
import yfinance as yf


def get_market_data(ticker: str, period: str = "1y") -> dict:
    """
    Fetch and process all market data needed for technical analysis.

    Returns dict with all computed indicators — ready for charting and LLM summary.
    """
    print(f"  [market_data] Fetching candles for {ticker}...")

    tk = yf.Ticker(ticker)

    # --- Fetch daily candles ---
    df = tk.history(period=period, interval="1d")
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")

    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # --- Fetch weekly candles ---
    df_weekly = tk.history(period=period, interval="1wk")
    df_weekly = df_weekly[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # --- Company info ---
    info = tk.info
    company_name = info.get("longName", ticker)
    market_cap = info.get("marketCap")
    current_price = float(df["Close"].iloc[-1])

    # --- Compute all indicators ---
    emas = _compute_emas(df)
    smas = _compute_smas(df)
    rsi_data = _compute_rsi(df)
    atr_data = _compute_atr(df)
    volume_analysis = _analyze_volume(df)
    patterns = _detect_chart_patterns(df)
    gaps = _detect_gaps(df)
    support, resistance = _find_support_resistance(df)
    swing_annotations = _find_swing_annotations(df)
    trend = _determine_trend(df, emas, smas)
    buy_sell = _compute_buy_sell_zones(df, support, resistance, emas, rsi_data, volume_analysis, trend)

    # --- Combined moving averages dict (for backward compat) ---
    moving_averages = _build_ma_dict(df, emas, smas)

    print(f"  [market_data] Got {len(df)} daily candles for {company_name}")

    return {
        "df": df,
        "df_weekly": df_weekly,
        "company_name": company_name,
        "market_cap": market_cap,
        "current_price": current_price,
        "moving_averages": moving_averages,
        "emas": emas,
        "smas": smas,
        "rsi": rsi_data,
        "atr": atr_data,
        "volume_analysis": volume_analysis,
        "patterns": patterns,
        "gaps": gaps,
        "support_levels": support,
        "resistance_levels": resistance,
        "swing_annotations": swing_annotations,
        "trend": trend,
        "buy_sell_zones": buy_sell,
        "last_candle": _analyze_last_candle(df),
    }


# ═══════════════════════════════════════════════════════════════════════
#  EMA / SMA
# ═══════════════════════════════════════════════════════════════════════

def _compute_emas(df: pd.DataFrame) -> dict:
    """Compute Exponential Moving Averages for 20, 50 periods."""
    close = df["Close"]
    emas = {}
    for period in [20, 50]:
        if len(df) >= period:
            series = close.ewm(span=period, adjust=False).mean()
            emas[f"ema{period}"] = series
            emas[f"ema{period}_current"] = float(series.iloc[-1])
    return emas


def _compute_smas(df: pd.DataFrame) -> dict:
    """Compute Simple Moving Averages for 150, 200 periods."""
    close = df["Close"]
    smas = {}
    for period in [150, 200]:
        if len(df) >= period:
            series = close.rolling(period).mean()
            smas[f"sma{period}"] = series
            smas[f"sma{period}_current"] = float(series.iloc[-1])
    return smas


def _build_ma_dict(df: pd.DataFrame, emas: dict, smas: dict) -> dict:
    """Build backward-compatible moving_averages dict."""
    close = df["Close"]
    current_price = float(close.iloc[-1])
    ma = {}

    # EMA 20, 50
    for period in [20, 50]:
        key = f"ema{period}_current"
        if key in emas:
            ma[f"ma{period}"] = emas[key]

    # SMA 150, 200
    for period in [150, 200]:
        key = f"sma{period}_current"
        if key in smas:
            ma[f"ma{period}"] = smas[key]

    # Also compute SMA 100 for backward compat
    if len(df) >= 100:
        ma["ma100"] = float(close.rolling(100).mean().iloc[-1])

    # Price position relative to key MAs
    for period in [20, 50, 200]:
        val = ma.get(f"ma{period}")
        if val:
            ma[f"price_vs_ma{period}"] = "above" if current_price > val else "below"
            ma[f"price_pct_from_ma{period}"] = round(
                (current_price - val) / val * 100, 2
            )

    # Price vs 150 MA
    val150 = ma.get("ma150")
    if val150:
        ma["price_vs_ma150"] = "above" if current_price > val150 else "below"
        ma["price_pct_from_ma150"] = round(
            (current_price - val150) / val150 * 100, 2
        )

    # Golden Cross / Death Cross (EMA50 vs SMA200)
    ma["golden_cross"] = False
    ma["death_cross"] = False
    if "ema50" in emas and "sma200" in smas:
        ema50_s = emas["ema50"]
        sma200_s = smas["sma200"]
        aligned = pd.DataFrame({"ema50": ema50_s, "sma200": sma200_s}).dropna()
        if len(aligned) >= 20:
            lookback = min(20, len(aligned))
            recent_diff = aligned["ema50"].iloc[-lookback:] - aligned["sma200"].iloc[-lookback:]
            if recent_diff.iloc[0] < 0 and recent_diff.iloc[-1] > 0:
                ma["golden_cross"] = True
            elif recent_diff.iloc[0] > 0 and recent_diff.iloc[-1] < 0:
                ma["death_cross"] = True

    return ma


# ═══════════════════════════════════════════════════════════════════════
#  RSI (14)
# ═══════════════════════════════════════════════════════════════════════

def _compute_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Compute RSI and detect overbought/oversold + divergences.

    Divergence: price makes new high but RSI does not (bearish),
                or price makes new low but RSI does not (bullish).
    """
    close = df["Close"]
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    current_rsi = float(rsi.iloc[-1])

    # Determine condition
    if current_rsi >= 70:
        condition = "overbought"
    elif current_rsi <= 30:
        condition = "oversold"
    elif current_rsi >= 60:
        condition = "bullish_momentum"
    elif current_rsi <= 40:
        condition = "bearish_momentum"
    else:
        condition = "neutral"

    # Divergence detection (last 30 bars)
    divergence = None
    if len(df) >= 30:
        recent_close = close.iloc[-30:]
        recent_rsi = rsi.iloc[-30:]

        # Find two highest peaks in price
        price_peaks = []
        rsi_at_peaks = []
        for i in range(2, len(recent_close) - 2):
            if (recent_close.iloc[i] > recent_close.iloc[i-1] and
                recent_close.iloc[i] > recent_close.iloc[i-2] and
                recent_close.iloc[i] > recent_close.iloc[i+1] and
                recent_close.iloc[i] > recent_close.iloc[i+2]):
                price_peaks.append(float(recent_close.iloc[i]))
                rsi_at_peaks.append(float(recent_rsi.iloc[i]))

        if len(price_peaks) >= 2:
            # Bearish divergence: higher price peak, lower RSI peak
            if price_peaks[-1] > price_peaks[-2] and rsi_at_peaks[-1] < rsi_at_peaks[-2]:
                divergence = "bearish"
            # Bullish divergence: lower price trough, higher RSI
            elif price_peaks[-1] < price_peaks[-2] and rsi_at_peaks[-1] > rsi_at_peaks[-2]:
                divergence = "bullish"

    return {
        "series": rsi,
        "current": current_rsi,
        "condition": condition,
        "divergence": divergence,
    }


# ═══════════════════════════════════════════════════════════════════════
#  ATR (14)
# ═══════════════════════════════════════════════════════════════════════

def _compute_atr(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Average True Range — measures volatility.
    ATR% = ATR / current_price — tells you how "wild" the stock is.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    current_atr = float(atr.iloc[-1]) if not atr.empty else 0
    current_price = float(df["Close"].iloc[-1])
    atr_pct = round(current_atr / current_price * 100, 2) if current_price > 0 else 0

    return {
        "series": atr,
        "current": round(current_atr, 2),
        "pct": atr_pct,
    }


# ═══════════════════════════════════════════════════════════════════════
#  VOLUME ANALYSIS — "lie detector"
# ═══════════════════════════════════════════════════════════════════════

def _analyze_volume(df: pd.DataFrame) -> dict:
    """
    Micha Stocks volume analysis:
    - High volume on green days = institutional buying (accumulation)
    - Low volume on pullbacks = healthy, no distribution
    - High volume on red days = distribution (selling pressure)
    """
    recent = df.tail(30).copy()
    avg_vol_50 = float(df["Volume"].tail(50).mean()) if len(df) >= 50 else float(df["Volume"].mean())

    recent["is_green"] = recent["Close"] > recent["Open"]
    green_days = recent[recent["is_green"]]
    red_days = recent[~recent["is_green"]]

    avg_green_vol = float(green_days["Volume"].mean()) if len(green_days) > 0 else 0
    avg_red_vol = float(red_days["Volume"].mean()) if len(red_days) > 0 else 0

    # Count high-volume green days (>1.3x average)
    high_vol_green = int((green_days["Volume"] > avg_vol_50 * 1.3).sum()) if len(green_days) > 0 else 0
    # Count high-volume red days
    high_vol_red = int((red_days["Volume"] > avg_vol_50 * 1.3).sum()) if len(red_days) > 0 else 0

    # Volume trend: is recent volume rising or declining?
    vol_10 = float(df["Volume"].tail(10).mean())
    vol_50 = avg_vol_50
    volume_trend = "rising" if vol_10 > vol_50 * 1.15 else ("declining" if vol_10 < vol_50 * 0.85 else "stable")

    # Determine accumulation/distribution signal
    if avg_green_vol > avg_red_vol * 1.3 and high_vol_green >= 3:
        signal = "accumulation"
        description = "Institutional buying detected — high volume on up days, low volume on pullbacks"
    elif avg_red_vol > avg_green_vol * 1.3 and high_vol_red >= 3:
        signal = "distribution"
        description = "Distribution detected — heavy selling on red days"
    elif volume_trend == "declining":
        signal = "quiet"
        description = "Volume drying up — watch for breakout"
    else:
        signal = "neutral"
        description = "No clear volume signal"

    return {
        "signal": signal,
        "description": description,
        "avg_vol_50": int(avg_vol_50),
        "avg_green_vol": int(avg_green_vol),
        "avg_red_vol": int(avg_red_vol),
        "high_vol_green_days": high_vol_green,
        "high_vol_red_days": high_vol_red,
        "volume_trend": volume_trend,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CHART PATTERNS — Cup & Handle, VCP, Flat Base
# ═══════════════════════════════════════════════════════════════════════

def _detect_chart_patterns(df: pd.DataFrame) -> list[str]:
    """
    Detect Micha Stocks style patterns:
    - Cup & Handle: rounding bottom + small pullback before breakout
    - VCP (Volatility Contraction Pattern): tightening price swings
    - Flat Base / Consolidation: sideways movement on declining volume
    """
    patterns = []
    n = len(df)

    # ── Cup & Handle ──────────────────────────────────────────────────
    if n >= 80:
        cup_seg = df.iloc[-80:-8]
        handle_seg = df.tail(8)
        cup_lows = cup_seg["Low"].values
        m = len(cup_lows)
        q = m // 4
        first_avg = float(np.mean(cup_lows[:q]))
        mid_avg = float(np.mean(cup_lows[m // 2 - q // 2: m // 2 + q // 2]))
        last_avg = float(np.mean(cup_lows[-q:]))
        cup_rim = float(cup_seg["High"].max())
        handle_low = float(handle_seg["Low"].min())
        pullback = (cup_rim - handle_low) / cup_rim if cup_rim > 0 else 1.0
        if (
            mid_avg < first_avg * 0.97
            and mid_avg < last_avg * 0.97
            and 0.02 <= pullback <= 0.15
        ):
            date_str = handle_seg.index[-1].strftime("%Y-%m-%d")
            patterns.append(
                f"Cup & Handle — breakout above ${cup_rim:.2f} (pullback: {pullback:.0%})"
            )

    # ── VCP (Volatility Contraction Pattern) ──────────────────────────
    # Look for 3+ contracting price swings in the last 60 bars
    if n >= 40:
        seg = df.tail(60) if n >= 60 else df.copy()
        highs = seg["High"].values
        lows = seg["Low"].values

        # Find swing ranges in 15-bar windows
        window = 15
        ranges = []
        for start in range(0, len(seg) - window, window // 2):
            end = min(start + window, len(seg))
            swing_range = float(max(highs[start:end]) - min(lows[start:end]))
            ranges.append(swing_range)

        if len(ranges) >= 3:
            contracting = all(ranges[i] < ranges[i - 1] * 0.85 for i in range(1, min(4, len(ranges))))
            if contracting:
                pivot = float(max(highs[-15:]))
                patterns.append(
                    f"VCP (Volatility Contraction) — tightening price, pivot at ${pivot:.2f}"
                )

    # ── Flat Base / Consolidation ─────────────────────────────────────
    # Price moves sideways (< 10% range) for 20+ days on declining volume
    if n >= 25:
        seg = df.tail(25)
        price_range = (float(seg["High"].max()) - float(seg["Low"].min()))
        mid_price = (float(seg["High"].max()) + float(seg["Low"].min())) / 2
        range_pct = price_range / mid_price if mid_price > 0 else 1

        if range_pct < 0.10:
            # Check for declining volume
            vol_first = float(seg["Volume"].iloc[:10].mean())
            vol_last = float(seg["Volume"].iloc[-10:].mean())
            if vol_last < vol_first * 0.85:
                patterns.append(
                    f"Flat Base — tight consolidation ({range_pct:.1%} range) with declining volume"
                )
            else:
                patterns.append(
                    f"Consolidation — sideways price action ({range_pct:.1%} range)"
                )

    # ── Rounding Bottom ───────────────────────────────────────────────
    if n >= 60:
        seg = df.tail(min(n, 100))
        lows = seg["Low"].values
        m = len(lows)
        q = m // 4
        first_avg = float(np.mean(lows[:q]))
        mid_avg = float(np.mean(lows[m // 2 - q // 2: m // 2 + q // 2]))
        last_avg = float(np.mean(lows[-q:]))
        if (
            mid_avg < first_avg * 0.97
            and mid_avg < last_avg * 0.97
            and abs(first_avg - last_avg) / first_avg < 0.08
        ):
            patterns.append("Rounding Bottom — gradual accumulation base forming")

    return patterns


# ═══════════════════════════════════════════════════════════════════════
#  GAP DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _detect_gaps(df: pd.DataFrame) -> list[dict]:
    """
    Detect significant price gaps in recent data.
    A gap up: today's low > yesterday's high
    A gap down: today's high < yesterday's low
    Only report gaps > 2% to avoid noise.
    """
    gaps = []
    recent = df.tail(30)

    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]
        date_str = recent.index[i].strftime("%Y-%m-%d")

        # Gap up
        if curr["Low"] > prev["High"]:
            gap_pct = (curr["Low"] - prev["High"]) / prev["High"] * 100
            if gap_pct >= 2:
                gaps.append({
                    "type": "gap_up",
                    "date": date_str,
                    "pct": round(gap_pct, 1),
                    "level": float(prev["High"]),  # gap fill level
                })

        # Gap down
        if curr["High"] < prev["Low"]:
            gap_pct = (prev["Low"] - curr["High"]) / prev["Low"] * 100
            if gap_pct >= 2:
                gaps.append({
                    "type": "gap_down",
                    "date": date_str,
                    "pct": round(gap_pct, 1),
                    "level": float(prev["Low"]),  # gap fill level
                })

    return gaps


# ═══════════════════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE — "areas of interest"
# ═══════════════════════════════════════════════════════════════════════

def _find_support_resistance(df: pd.DataFrame, window: int = 10) -> tuple[list, list]:
    """
    Find support and resistance using swing detection + volume confirmation.
    Clusters nearby levels. Returns max 3 of each (clean chart).
    """
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    n = len(df)
    avg_vol = float(np.mean(volumes))

    swing_lows = []
    swing_highs = []

    for i in range(window, n - window):
        if lows[i] == min(lows[i - window: i + window + 1]):
            # Weight by volume — higher volume = more significant level
            vol_weight = volumes[i] / avg_vol if avg_vol > 0 else 1
            swing_lows.append((float(lows[i]), vol_weight, i))
        if highs[i] == max(highs[i - window: i + window + 1]):
            vol_weight = volumes[i] / avg_vol if avg_vol > 0 else 1
            swing_highs.append((float(highs[i]), vol_weight, i))

    def cluster_levels(levels: list, threshold_pct: float = 0.02) -> list:
        """Cluster nearby levels, keeping the one with highest volume weight."""
        if not levels:
            return []
        levels = sorted(levels, key=lambda x: x[0])
        clustered = [levels[0]]
        for level, weight, idx in levels[1:]:
            if abs(level - clustered[-1][0]) / clustered[-1][0] <= threshold_pct:
                # Keep the one with higher volume weight
                if weight > clustered[-1][1]:
                    clustered[-1] = (level, weight, idx)
            else:
                clustered.append((level, weight, idx))
        # Sort by volume weight (most significant first), then take top ones
        clustered.sort(key=lambda x: x[1], reverse=True)
        return [round(c[0], 2) for c in clustered]

    current_price = float(df["Close"].iloc[-1])
    support = [s for s in cluster_levels(swing_lows) if s < current_price][:3]
    resistance = [r for r in cluster_levels(swing_highs) if r > current_price][:3]

    support.sort(reverse=True)   # nearest first
    resistance.sort()             # nearest first

    return support, resistance


def _find_swing_annotations(df: pd.DataFrame, window: int = 12) -> list[dict]:
    """
    Find key swing high/low points for chart annotation (price labels).
    Like Micha Stocks labeling key pivots on the chart.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    annotations = []

    for i in range(window, n - window):
        date = df.index[i]
        if highs[i] == max(highs[i - window: i + window + 1]):
            annotations.append({
                "type": "high",
                "date": date,
                "price": round(float(highs[i]), 2),
            })
        if lows[i] == min(lows[i - window: i + window + 1]):
            annotations.append({
                "type": "low",
                "date": date,
                "price": round(float(lows[i]), 2),
            })

    # Keep only the most significant ones (last 8)
    return annotations[-8:]


# ═══════════════════════════════════════════════════════════════════════
#  TREND DETERMINATION
# ═══════════════════════════════════════════════════════════════════════

def _determine_trend(df: pd.DataFrame, emas: dict, smas: dict) -> dict:
    """
    Determine trend using MA alignment and price action.

    Strong uptrend:  Price > EMA20 > EMA50 > SMA150 > SMA200 (all aligned)
    Moderate uptrend: Price above most MAs
    Downtrend:       Price below most MAs, MAs curling down
    """
    current_price = float(df["Close"].iloc[-1])

    ma_values = []
    for key in ["ema20_current", "ema50_current"]:
        if key in emas:
            ma_values.append((key, emas[key]))
    for key in ["sma150_current", "sma200_current"]:
        if key in smas:
            ma_values.append((key, smas[key]))

    above_count = sum(1 for _, v in ma_values if current_price > v)
    total = len(ma_values)

    # Check MA alignment (perfect order)
    values_only = [v for _, v in ma_values]
    perfectly_aligned_bull = all(values_only[i] >= values_only[i + 1] for i in range(len(values_only) - 1)) if len(values_only) >= 3 else False
    price_above_all = current_price > max(values_only) if values_only else False

    # Check if price is extended (too far from EMA20)
    extended = False
    if "ema20_current" in emas:
        dist_from_ema20 = (current_price - emas["ema20_current"]) / emas["ema20_current"] * 100
        if dist_from_ema20 > 15:
            extended = True

    # EMA20 acting as dynamic support? (price bouncing off it)
    ema20_dynamic_support = False
    if "ema20" in emas:
        recent_lows = df["Low"].tail(10)
        ema20_recent = emas["ema20"].tail(10)
        touches = sum(1 for l, e in zip(recent_lows, ema20_recent)
                      if abs(l - e) / e < 0.015)  # within 1.5%
        if touches >= 2:
            ema20_dynamic_support = True

    if price_above_all and perfectly_aligned_bull:
        direction = "strong_uptrend"
        strength = "strong"
    elif above_count >= 3 and total >= 4:
        direction = "uptrend"
        strength = "moderate"
    elif above_count <= 1 and total >= 3:
        direction = "downtrend"
        strength = "strong" if above_count == 0 else "moderate"
    else:
        direction = "sideways"
        strength = "weak"

    return {
        "direction": direction,
        "strength": strength,
        "above_ma_count": above_count,
        "total_mas": total,
        "perfectly_aligned": perfectly_aligned_bull and price_above_all,
        "extended": extended,
        "ema20_dynamic_support": ema20_dynamic_support,
    }


# ═══════════════════════════════════════════════════════════════════════
#  BUY / SELL ZONES
# ═══════════════════════════════════════════════════════════════════════

def _compute_buy_sell_zones(
    df: pd.DataFrame,
    support: list,
    resistance: list,
    emas: dict,
    rsi_data: dict,
    volume_analysis: dict,
    trend: dict,
) -> dict:
    """
    Hardcoded buy/sell logic — Micha Stocks style:

    BUY ZONE (look for entries):
      - Near primary support OR EMA20/50 (dynamic support)
      - RSI not overbought (< 65 ideal, < 70 acceptable)
      - Volume shows accumulation or quiet (not distribution)
      - Trend is up or sideways (not strong downtrend)

    SELL ZONE (take profits / manage risk):
      - Near primary resistance
      - RSI overbought (> 70) or bearish divergence
      - Price extended far above EMA20
      - Volume shows distribution
    """
    current_price = float(df["Close"].iloc[-1])
    buy_zone = None
    sell_zone = None
    buy_reasons = []
    sell_reasons = []

    # ── BUY ZONE ──────────────────────────────────────────────────────
    # Primary: nearest support level
    if support:
        buy_zone = support[0]
        buy_reasons.append(f"Primary support at ${support[0]:.2f}")

    # Alternative: EMA20 as dynamic support if closer
    if "ema20_current" in emas and trend["direction"] in ("uptrend", "strong_uptrend"):
        ema20_val = emas["ema20_current"]
        if ema20_val < current_price:
            if buy_zone is None or abs(ema20_val - current_price) < abs(buy_zone - current_price):
                buy_zone = round(ema20_val, 2)
                buy_reasons.insert(0, f"EMA20 dynamic support at ${ema20_val:.2f}")

    # Confidence modifiers
    if rsi_data["condition"] == "oversold":
        buy_reasons.append("RSI oversold — watch for reversal")
    if volume_analysis["signal"] == "accumulation":
        buy_reasons.append("Volume confirms accumulation")
    if volume_analysis["signal"] == "quiet":
        buy_reasons.append("Volume quiet — potential breakout setup")

    # ── SELL ZONE ─────────────────────────────────────────────────────
    if resistance:
        sell_zone = resistance[0]
        sell_reasons.append(f"Primary resistance at ${resistance[0]:.2f}")

    if rsi_data["condition"] == "overbought":
        sell_reasons.append("RSI overbought (>70)")
    if rsi_data["divergence"] == "bearish":
        sell_reasons.append("Bearish RSI divergence")
    if trend["extended"]:
        sell_reasons.append("Price extended far above EMA20")
    if volume_analysis["signal"] == "distribution":
        sell_reasons.append("Volume shows distribution")

    # ── OVERALL ACTION ────────────────────────────────────────────────
    if trend["direction"] == "downtrend" and volume_analysis["signal"] == "distribution":
        action = "avoid"
        action_reason = "Downtrend with distribution — stay away"
    elif (trend["direction"] in ("uptrend", "strong_uptrend") and
          rsi_data["condition"] not in ("overbought",) and
          volume_analysis["signal"] != "distribution"):
        if rsi_data["condition"] == "oversold" or trend.get("ema20_dynamic_support"):
            action = "buy_on_pullback"
            action_reason = "Uptrend — buy on pullback to support"
        else:
            action = "watch_for_entry"
            action_reason = "Uptrend — wait for pullback to buy zone"
    elif rsi_data["condition"] == "overbought" or trend["extended"]:
        action = "take_profits"
        action_reason = "Consider taking partial profits"
    else:
        action = "hold"
        action_reason = "No clear edge — wait for setup"

    return {
        "buy_zone": buy_zone,
        "buy_reasons": buy_reasons,
        "sell_zone": sell_zone,
        "sell_reasons": sell_reasons,
        "action": action,
        "action_reason": action_reason,
    }


# ═══════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════

def _analyze_last_candle(df: pd.DataFrame) -> dict:
    """Analyze the anatomy of the most recent candle."""
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


def format_candles_for_llm(df: pd.DataFrame, n_candles: int = 20) -> str:
    """Format recent candle data as a compact string for LLM prompt (reduced from 30 to 20)."""
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
