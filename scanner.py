"""
scanner.py
==========
Scans favorite tickers for buy-zone alerts during trading hours.

BUY ZONE LOGIC:
  The "buy zone" is defined by the support levels from the most recent
  TechnicalReport analysis. A ticker is in the buy zone when:
    - Price is within BUY_ZONE_THRESHOLD_PCT (default 3%) above primary support
    - OR price has dropped below primary support (even stronger signal)

STALENESS:
  If the last analysis is older than MAX_ANALYSIS_AGE_DAYS (default 3),
  a fresh analysis is automatically triggered before scanning.

TRADING HOURS:
  US market: 9:30 AM – 4:00 PM Eastern Time, Monday–Friday.
  The scanner warns (but still runs) outside trading hours.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import yfinance as yf

from analysis_store import load_analysis, get_analysis_age_days, save_analysis

# ── Configuration ──────────────────────────────────────────────────────────

# Tight thresholds — alert fires when price is THIS close to the key level.
# 1.5% means: if support is $100, alert triggers at $101.50 or below.
# Set tight so alerts mean "price is AT the line", not "approaching the zone".
BUY_ZONE_THRESHOLD_PCT  = 1.5  # Price within X% above support = at buy line
SELL_ZONE_THRESHOLD_PCT = 1.5  # Price within X% below resistance = at sell line

MAX_ANALYSIS_AGE_DAYS = 3      # Re-analyze if older than this
EASTERN = ZoneInfo("America/New_York")

FAVORITES_FILE   = "data/favorites.json"
ALERT_STATE_FILE = "data/alert_state.json"   # tracks last zone status per ticker


# ── Data classes ───────────────────────────────────────────────────────────

class ScanResult:
    """Result of scanning a single ticker."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.current_price: Optional[float] = None
        # Buy zone (support)
        self.support_levels: list[float] = []
        self.primary_support: Optional[float] = None
        self.distance_to_support_pct: Optional[float] = None
        self.in_buy_zone: bool = False
        self.below_support: bool = False
        # Sell zone (resistance)
        self.resistance_levels: list[float] = []
        self.primary_resistance: Optional[float] = None
        self.distance_to_resistance_pct: Optional[float] = None
        self.in_sell_zone: bool = False
        self.above_resistance: bool = False
        # Analysis metadata
        self.verdict: Optional[str] = None
        self.composite_score: Optional[float] = None
        self.analysis_date: Optional[str] = None
        self.analysis_refreshed: bool = False
        self.error: Optional[str] = None
        self.price_target: Optional[str] = None

    @property
    def zone_status(self) -> str:
        """Human-readable zone status."""
        if self.error:
            return "ERROR"
        if self.below_support:
            return "BELOW SUPPORT"
        if self.in_buy_zone:
            return "IN BUY ZONE"
        if self.above_resistance:
            return "ABOVE RESISTANCE"
        if self.in_sell_zone:
            return "IN SELL ZONE"
        return "BETWEEN ZONES"

    @property
    def has_alert(self) -> bool:
        """True if any zone was triggered (buy or sell)."""
        return self.in_buy_zone or self.below_support or self.in_sell_zone or self.above_resistance

    @property
    def alert_message(self) -> str:
        if self.error:
            return f"[ERROR] {self.ticker}: {self.error}"

        if not self.has_alert:
            sup_dist = f"{self.distance_to_support_pct:+.1f}%" if self.distance_to_support_pct is not None else "N/A"
            res_dist = f"{self.distance_to_resistance_pct:+.1f}%" if self.distance_to_resistance_pct is not None else "N/A"
            return (
                f"{self.ticker}: ${self.current_price:.2f} | "
                f"Support: {sup_dist} | Resistance: {res_dist} | BETWEEN ZONES"
            )

        lines = [f"🚨 {self.ticker} — {self.zone_status}!"]
        lines.append(f"Current Price: ${self.current_price:.2f}")
        if self.primary_support:
            lines.append(f"Buy Zone (Support): ${self.primary_support:.2f} ({self.distance_to_support_pct:+.1f}%)")
        if self.primary_resistance:
            lines.append(f"Sell Zone (Resistance): ${self.primary_resistance:.2f} ({self.distance_to_resistance_pct:+.1f}%)")
        lines.append(f"Verdict: {self.verdict} ({self.composite_score:.1f}/10)")
        lines.append(f"Target: {self.price_target or 'N/A'}")
        lines.append(f"Analysis: {self.analysis_date}")
        return "\n".join(lines)


# ── Alert state tracking ──────────────────────────────────────────────────
# Persists the last zone status for each ticker so Telegram only fires
# when the price FIRST ENTERS a buy/sell zone — not on every scan refresh.

def _load_alert_state() -> dict:
    """Load persisted zone-status dict from disk."""
    if os.path.exists(ALERT_STATE_FILE):
        try:
            with open(ALERT_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_alert_state(state: dict) -> None:
    os.makedirs("data", exist_ok=True)
    try:
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


def check_and_update_zone_state(ticker: str, new_status: str) -> bool:
    """
    Check if this ticker just ENTERED a buy/sell zone (state changed).
    Updates the persisted state file.

    Returns True only when:
      - The previous status was BETWEEN ZONES (or first scan)
      - AND the new status is IN BUY ZONE / BELOW SUPPORT / IN SELL ZONE / ABOVE RESISTANCE

    This means Telegram fires exactly once per zone entry, not on every scan.
    """
    ACTIONABLE = {"IN BUY ZONE", "BELOW SUPPORT", "IN SELL ZONE", "ABOVE RESISTANCE"}
    state = _load_alert_state()
    old_status = state.get(ticker, "BETWEEN ZONES")

    # Always persist the new status
    state[ticker] = new_status
    _save_alert_state(state)

    # Fire alert only on ENTRY into a zone (from outside)
    just_entered = (old_status not in ACTIONABLE) and (new_status in ACTIONABLE)
    return just_entered


# ── Core scanning logic ───────────────────────────────────────────────────

def is_trading_hours() -> tuple[bool, str]:
    """Check if US stock market is currently open."""
    now_et = datetime.now(EASTERN)

    if now_et.weekday() >= 5:
        return False, f"Weekend ({now_et.strftime('%A')})"

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if now_et < market_open:
        return False, f"Pre-market ({now_et.strftime('%H:%M')} ET, opens 9:30)"
    if now_et > market_close:
        return False, f"After-hours ({now_et.strftime('%H:%M')} ET, closed at 16:00)"

    return True, f"Market open ({now_et.strftime('%H:%M')} ET)"


def get_live_price(ticker: str) -> Optional[float]:
    """
    Fetch the current/latest price for a ticker via yfinance.
    For the DEMO ticker, returns a hardcoded price in the buy zone.
    """
    if ticker == "DEMO":
        return _get_demo_price()

    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price:
            return float(price)
        # Fallback: latest close from history
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def scan_ticker(ticker: str, auto_refresh: bool = True) -> ScanResult:
    """
    Scan a single ticker for buy-zone proximity.

    Args:
        ticker:       Stock symbol
        auto_refresh: If True, re-run analysis when stale (> 3 days old)

    Returns:
        ScanResult with buy-zone detection details.
    """
    result = ScanResult(ticker)

    # ── Load or refresh analysis ──────────────────────────────────────
    age_days = get_analysis_age_days(ticker)

    if age_days is None and ticker == "DEMO":
        # Auto-create demo analysis if it doesn't exist
        create_demo_analysis()
        age_days = 0

    if age_days is None:
        if auto_refresh:
            try:
                _run_fresh_analysis(ticker)
                result.analysis_refreshed = True
            except Exception as e:
                result.error = f"No analysis found and refresh failed: {e}"
                return result
        else:
            result.error = "No analysis data. Run analysis first."
            return result

    elif age_days > MAX_ANALYSIS_AGE_DAYS and auto_refresh and ticker != "DEMO":
        try:
            _run_fresh_analysis(ticker)
            result.analysis_refreshed = True
        except Exception as e:
            # Use stale data as fallback
            pass

    # ── Read analysis data ────────────────────────────────────────────
    data = load_analysis(ticker)
    if not data:
        result.error = "Analysis file missing."
        return result

    result.analysis_date = data.get("analysis_date", "Unknown")
    result.verdict = data.get("final", {}).get("verdict")
    result.composite_score = data.get("final", {}).get("composite_score")
    result.price_target = data.get("final", {}).get("price_target")
    result.support_levels = data.get("technical", {}).get("support_levels", [])
    result.resistance_levels = data.get("technical", {}).get("resistance_levels", [])

    if not result.support_levels and not result.resistance_levels:
        result.error = "No support/resistance levels in analysis."
        return result

    if result.support_levels:
        result.primary_support = result.support_levels[0]
    if result.resistance_levels:
        result.primary_resistance = result.resistance_levels[0]

    # ── Get live price ────────────────────────────────────────────────
    result.current_price = get_live_price(ticker)
    if result.current_price is None:
        result.error = "Could not fetch current price."
        return result

    price = result.current_price

    # ── Buy zone detection (near support) ─────────────────────────────
    if result.primary_support:
        support = result.primary_support
        result.distance_to_support_pct = ((price - support) / support) * 100
        result.below_support = price < support
        result.in_buy_zone = (
            result.below_support or
            result.distance_to_support_pct <= BUY_ZONE_THRESHOLD_PCT
        )

    # ── Sell zone detection (near resistance) ─────────────────────────
    if result.primary_resistance:
        resistance = result.primary_resistance
        result.distance_to_resistance_pct = ((price - resistance) / resistance) * 100
        result.above_resistance = price > resistance
        result.in_sell_zone = (
            result.above_resistance or
            abs(result.distance_to_resistance_pct) <= SELL_ZONE_THRESHOLD_PCT
        )

    return result


def scan_favorites(auto_refresh: bool = True) -> list[ScanResult]:
    """
    Scan all favorite tickers for buy-zone alerts.

    Returns:
        List of ScanResult objects, one per favorite ticker.
    """
    favorites = load_favorites()
    results = []
    for ticker in favorites:
        result = scan_ticker(ticker, auto_refresh=auto_refresh)
        results.append(result)
    return results


def load_favorites() -> list[str]:
    """Load favorite tickers from data/favorites.json."""
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


# ── Demo ticker ────────────────────────────────────────────────────────────

def _get_demo_price() -> float:
    """Return a fake price for DEMO that sits right in the buy zone."""
    return 101.50  # Primary support is at 100.00 → 1.5% above = IN buy zone


def create_demo_analysis() -> str:
    """
    Create a fake analysis for ticker 'DEMO' that is in the buy zone.
    This lets you test the full scan → alert pipeline without real market data.

    Demo setup:
      - Current price: $101.50
      - Primary support: $100.00 (buy zone = within 3%)
      - Verdict: BUY (7.5/10)
      - The scan will detect this as IN BUY ZONE
    """
    from models.report import (
        TechnicalReport, MovingAverages,
        FundamentalReport, FundamentalMetric,
        SentimentReport, FinalReport,
    )

    tech = TechnicalReport(
        ticker="DEMO",
        current_price=101.50,
        trend="Bullish",
        trend_strength="Moderate",
        key_patterns=["Cup & Handle — breakout above $105.00 (pullback: 8%)"],
        moving_averages=MovingAverages(
            ma20=103.0, ma50=98.5, ma100=95.0, ma150=92.0, ma200=90.0,
            price_vs_ma20="below", price_vs_ma50="above",
            price_vs_ma150="above", price_vs_ma200="above",
            golden_cross=True, death_cross=False,
        ),
        rsi_value=55.0, rsi_condition="neutral", rsi_divergence=None,
        atr_value=3.2, atr_pct=3.15,
        volume_signal="accumulation", volume_description="Institutional buying detected",
        support_levels=[100.00, 95.00, 90.00],
        resistance_levels=[105.00, 110.00, 118.00],
        buy_zone=100.00, buy_reasons=["Primary support at $100.00", "EMA20 dynamic support"],
        sell_zone=105.00, sell_reasons=["Primary resistance at $105.00"],
        action="buy_on_pullback", action_reason="Uptrend — buy on pullback to support",
        score=7.0,
        summary="DEMO is testing support at $100 with accumulation volume.",
        short_term_outlook="Likely bounce off $100 support toward $105 resistance.",
    )

    n = FundamentalMetric(grade="B", comment="Demo data")
    fund = FundamentalReport(
        ticker="DEMO",
        company_name="Demo Corp (Test Ticker)",
        sector="Technology",
        industry="Software",
        revenue_growth_yoy=FundamentalMetric(value=12.0, grade="B", comment="12% YoY"),
        net_income_margin=FundamentalMetric(value=18.0, grade="A", comment="18% margin"),
        pe_ratio=FundamentalMetric(value=22.0, grade="B", comment="22x P/E"),
        eps_growth=FundamentalMetric(value=15.0, grade="B", comment="15% growth"),
        return_on_equity=FundamentalMetric(value=20.0, grade="A", comment="20% ROE"),
        debt_to_equity=FundamentalMetric(value=0.4, grade="A", comment="0.4x D/E"),
        free_cash_flow=FundamentalMetric(value=500.0, grade="A", comment="$500M FCF"),
        market_cap_billions=25.0,
        revenue_ttm_billions=5.0,
        score=7.8,
        summary="Demo Corp has solid fundamentals with strong margins and low debt.",
        key_strengths=["High ROE", "Low debt", "Strong FCF"],
        key_concerns=["Demo data only", "Not a real company"],
    )

    sent = SentimentReport(
        ticker="DEMO",
        company_name="Demo Corp (Test Ticker)",
        articles_analyzed=10,
        date_range="2026-03-25 to 2026-04-01",
        overall_sentiment="Bullish",
        sentiment_score=0.6,
        key_themes=["Buy zone test", "Scanner demo"],
        upcoming_catalysts=["Q1 earnings in 2 weeks"],
        risks_mentioned=["This is a demo ticker"],
        sources_used=["Demo News"],
        score=7.5,
        summary="Bullish sentiment from demo news sources for testing purposes.",
    )

    final = FinalReport(
        ticker="DEMO",
        company_name="Demo Corp (Test Ticker)",
        report_date=datetime.now().strftime("%Y-%m-%d"),
        technical_score=7.0,
        fundamental_score=7.8,
        sentiment_score=7.5,
        composite_score=7.5,
        verdict="BUY",
        confidence_pct=72.0,
        time_horizon="3-6 months",
        current_price=101.50,
        price_target="$110-115",
        analyst_thesis=(
            "Demo Corp is testing its primary support at $100 with bullish reversal patterns. "
            "Strong fundamentals (18% margins, 20% ROE) support a buy thesis near support. "
            "This is a DEMO ticker created to test the scanner buy-zone alert system."
        ),
        key_opportunities=["Price near support = good entry", "Golden cross on MAs"],
        key_risks=["This is a demo ticker", "Not real market data"],
        bull_case="Bounce off $100 support toward $110-115 target",
        bear_case="Break below $100 support could lead to $95 retest",
        watch_for=["$100 support hold", "Volume confirmation", "Q1 earnings"],
    )

    path = save_analysis("DEMO", tech, fund, sent, final)
    return path


# ── Fresh analysis runner ──────────────────────────────────────────────────

def _run_fresh_analysis(ticker: str) -> None:
    """Run a fresh full analysis and save it."""
    from orchestrator import run_analysis
    run_analysis(
        ticker=ticker,
        period="1y",
        stream_output=False,
        llm_provider="api",
        open_browser=False,  # scanner runs in a loop — no tab spam.
    )
