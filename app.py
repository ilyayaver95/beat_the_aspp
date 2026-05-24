"""
app.py
======
Streamlit web UI for Beat the ASPP.

USAGE:
  streamlit run app.py

FEATURES:
  - Type any ticker → run full AI analysis
  - Save tickers to a Favorites list (persisted to data/favorites.json)
  - One-click re-analysis from favorites
  - View cached past analyses without re-running (date shown in sidebar)
  - Choose LLM provider: Anthropic API or local Ollama
  - Analysis result is embedded inline as a scrollable HTML report
"""

import json
import os
import glob
import traceback
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # loads .env locally

import streamlit as st
import streamlit.components.v1 as components
from cost_tracker import tracker as _cost_tracker, predict_analysis_cost

# Streamlit Cloud stores secrets in st.secrets, not os.environ.
# Sync them so libraries like the Anthropic SDK can read them normally.
_secrets_sync_status = "no st.secrets access"
_secrets_sync_keys = []
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
            _secrets_sync_keys.append(_k)
    _secrets_sync_status = f"synced {len(_secrets_sync_keys)} keys: {_secrets_sync_keys}"
except Exception as _e:
    _secrets_sync_status = f"st.secrets failed: {type(_e).__name__}: {_e}"

# Snapshot any deployer-provided keys ONCE at startup. After this, env vars
# are managed per-session from st.session_state via _apply_user_keys().
_DEPLOYER_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_DEPLOYER_GROQ_KEY      = os.environ.get("GROQ_API_KEY", "")


def _apply_user_keys() -> None:
    """
    Apply this session's API keys to os.environ.

    Precedence: user-pasted key (st.session_state) > deployer key (snapshot above).
    Called at the top of every script rerun so each user sees their own keys.
    """
    user_anth = st.session_state.get("user_anthropic_key", "").strip()
    user_groq = st.session_state.get("user_groq_key", "").strip()

    eff_anth = user_anth or _DEPLOYER_ANTHROPIC_KEY
    eff_groq = user_groq or _DEPLOYER_GROQ_KEY

    if eff_anth:
        os.environ["ANTHROPIC_API_KEY"] = eff_anth
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)

    if eff_groq:
        os.environ["GROQ_API_KEY"] = eff_groq
    else:
        os.environ.pop("GROQ_API_KEY", None)


_apply_user_keys()

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beat the ASPP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

FAVORITES_FILE = "data/favorites.json"


# ── Helpers ────────────────────────────────────────────────────────
def load_favorites() -> list[str]:
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_favorites(favs: list[str]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
        json.dump(favs, f)


def get_saved_reports(ticker: str) -> list[tuple[str, str]]:
    """Return [(date_str, html_path), ...] sorted newest-first for the given ticker."""
    matches = glob.glob(f"reports/{ticker}_*.html")
    results = []
    for path in matches:
        filename = os.path.basename(path)
        # Expected format: TICKER_YYYY-MM-DD.html
        date_part = filename[len(ticker) + 1:].replace(".html", "")
        # Only keep entries whose date part looks like a date
        if len(date_part) == 10 and date_part[4] == "-" and date_part[7] == "-":
            results.append((date_part, path))
    return sorted(results, reverse=True)  # newest first


def run_analysis_cached(ticker: str, provider: str, model: str, period: str = "1y") -> tuple:
    """Run the full analysis and return (FinalReport, html_path)."""
    from orchestrator import run_analysis
    report = run_analysis(
        ticker=ticker,
        period=period,
        stream_output=False,
        llm_provider=provider,
        llm_model=model if provider in ("ollama", "groq") else None,
    )
    date_str = datetime.now().strftime("%Y-%m-%d")
    pattern = f"reports/{ticker}_{date_str}.html"
    matches = glob.glob(pattern)
    html_path = matches[0] if matches else None
    return report, html_path


def get_cache_info(ticker: str) -> dict | None:
    """
    Check if a recent analysis JSON exists for this ticker.
    Returns dict with age_hours and html_path if < 24h old, else None.
    """
    import json as _json
    cache_file = f"data/analyses/{ticker}_latest.json"
    if not os.path.exists(cache_file):
        return None
    age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
    if age_hours >= 24:
        return None
    # Find the most recent HTML report for this ticker
    saved = get_saved_reports(ticker)
    html_path = saved[0][1] if saved else None
    try:
        with open(cache_file, encoding="utf-8") as f:
            data = _json.load(f)
        return {
            "age_hours": age_hours,
            "html_path": html_path,
            "verdict": data.get("verdict", "?"),
            "score": data.get("composite_score", "?"),
        }
    except Exception:
        return None


def _render_html_report(html_path: str, header_msg: str = "", header_type: str = "info") -> None:
    """Load and render an HTML report file with an optional header banner."""
    if not html_path or not os.path.exists(html_path):
        st.warning("HTML report file not found. Check the `reports/` directory.")
        return
    if header_msg:
        getattr(st, header_type)(header_msg)
    with open(html_path, encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=4200, scrolling=True)


# ── Sidebar — Favorites ────────────────────────────────────────────
with st.sidebar:
    # ── API Keys (Bring Your Own) ──────────────────────────────────────
    _has_deployer_anth = bool(_DEPLOYER_ANTHROPIC_KEY)
    _has_deployer_groq = bool(_DEPLOYER_GROQ_KEY)
    _has_user_anth = bool(st.session_state.get("user_anthropic_key", "").strip())
    _has_user_groq = bool(st.session_state.get("user_groq_key", "").strip())

    _keys_needed = not (_has_deployer_groq or _has_user_groq or _has_deployer_anth or _has_user_anth)
    with st.expander("🔑 API Keys", expanded=_keys_needed):
        # Groq is built in — show status, allow override
        if _has_user_groq:
            st.success("Groq: using **your** key.")
        elif _has_deployer_groq:
            st.success("Groq: **ready to use** (no key needed).")
        else:
            st.warning("Groq: no key configured.")
            st.text_input(
                "Groq API Key (free)",
                key="user_groq_key",
                type="password",
                placeholder="gsk_...",
                help="Free key at https://console.groq.com/keys",
                on_change=_apply_user_keys,
            )

        st.divider()
        st.caption("**Optional — Anthropic (higher quality, paid):**")

        if _has_user_anth:
            st.success("Anthropic: using **your** key — your account is being charged.")
        elif _has_deployer_anth:
            st.info("Anthropic: using deployer's key (free for you).")
        else:
            st.caption("Anthropic: no key set — Anthropic mode will fail.")

        st.text_input(
            "Anthropic API Key",
            key="user_anthropic_key",
            type="password",
            placeholder="sk-ant-api03-...",
            help="Get one at https://console.anthropic.com/settings/keys",
            on_change=_apply_user_keys,
        )

    st.markdown("## ⭐ Favorites")
    favorites = load_favorites()

    # Initialize scanning set in session state
    if "scanning_tickers" not in st.session_state:
        st.session_state["scanning_tickers"] = set()

    if not favorites:
        st.caption("No favorites yet. Add a ticker after running analysis.")
    else:
        for fav in list(favorites):
            saved = get_saved_reports(fav)
            latest_date = saved[0][0] if saved else None
            is_scanning = fav in st.session_state["scanning_tickers"]

            col_btn, col_scan, col_del = st.columns([3, 1, 1])

            if col_btn.button(fav, key=f"fav_{fav}", use_container_width=True):
                st.session_state["selected_ticker"] = fav
                st.session_state.pop("view_cache", None)
                st.rerun()

            # Toggle scan on/off — green when active, default when off
            scan_label = "✅" if is_scanning else "🔍"
            scan_help = f"Stop scanning {fav}" if is_scanning else f"Scan {fav} for buy/sell zones"
            scan_type = "primary" if is_scanning else "secondary"
            if col_scan.button(scan_label, key=f"scan_{fav}", help=scan_help, type=scan_type):
                if is_scanning:
                    st.session_state["scanning_tickers"].discard(fav)
                    # Remove cached result for this ticker
                    st.session_state.get("scan_results", {}).pop(fav, None)
                else:
                    st.session_state["scanning_tickers"].add(fav)
                    st.session_state["scan_single"] = fav
                st.rerun()

            if latest_date:
                col_btn.caption(f"Last: {latest_date}")

            if col_del.button("✕", key=f"del_{fav}"):
                favorites.remove(fav)
                save_favorites(favorites)
                st.session_state["scanning_tickers"].discard(fav)
                st.rerun()

    # ── Scan All / Stop All ───────────────────────────────────────
    st.divider()
    active_scans = st.session_state["scanning_tickers"]
    if favorites:
        col_scan_all, col_stop_all = st.columns(2)
        with col_scan_all:
            if st.button("🔍 Scan All", use_container_width=True):
                st.session_state["scanning_tickers"] = set(favorites)
                st.session_state["scan_all"] = True
                st.rerun()
        with col_stop_all:
            if st.button("⏹ Stop All", use_container_width=True,
                         disabled=len(active_scans) == 0):
                st.session_state["scanning_tickers"] = set()
                st.session_state.pop("scan_results", None)
                st.rerun()

    if active_scans:
        st.caption(f"Scanning: {', '.join(sorted(active_scans))}")

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "AI-Powered Stock Evaluator using three parallel agents:\n"
        "Technical · Fundamental · Sentiment"
    )


# ── Main area ──────────────────────────────────────────────────────
st.title("📈 Beat the ASPP")
st.caption("AI-Powered Stock Evaluation · Technical + Fundamental + Sentiment")

# ── Cost Tracker Panel ─────────────────────────────────────────────
# Pure helpers — defined outside fragment so they aren't re-created each tick.
def _cost_color(cost: float) -> str:
    if cost < 0.10:
        return "#2ecc71"
    if cost < 1.00:
        return "#f39c12"
    return "#e74c3c"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _render_cost_col(label: str, stats: dict, icon: str) -> None:
    cost = stats["cost"]
    color = _cost_color(cost)
    st.markdown(
        f"<div style='text-align:center;padding:6px 0 2px 0'>"
        f"<span style='font-size:1.4rem'>{icon}</span>&nbsp;"
        f"<strong style='font-size:.95rem'>{label}</strong></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='text-align:center;margin-bottom:4px'>"
        f"<span style='font-size:1.8rem;font-weight:700;color:{color}'>"
        f"${cost:.4f}</span></div>",
        unsafe_allow_html=True,
    )
    if stats["calls"] == 0:
        st.caption("No API calls recorded")
    else:
        st.caption(
            f"**{stats['calls']}** calls · **{_fmt_tokens(stats['total'])}** tokens  \n"
            f"↑ {_fmt_tokens(stats['input'])} in · ↓ {_fmt_tokens(stats['output'])} out"
        )
        if stats.get("cache_read", 0) or stats.get("cache_write", 0):
            st.caption(
                f"Cache: {_fmt_tokens(stats['cache_read'])} read · "
                f"{_fmt_tokens(stats['cache_write'])} write"
            )


@st.fragment(run_every=5)
def _cost_tracker_panel() -> None:
    """
    Auto-refreshing cost tracker panel.
    st.fragment(run_every=5) re-executes this block every 5 seconds
    independently of the rest of the page — no full-page flicker.
    """
    sess  = _cost_tracker.get_session_stats()
    h24   = _cost_tracker.get_24h_stats()
    total = _cost_tracker.get_total_stats()
    last_run   = st.session_state.get("last_run_cost")
    last_tokens = st.session_state.get("last_run_tokens", {})

    title = (
        f"💰 API Cost Tracker — Session: **${sess['cost']:.4f}**"
        + (f" · Last run: **${last_run:.4f}**" if last_run else "")
    )
    with st.expander(title, expanded=False):
        col_s, col_24, col_tot, col_live = st.columns(4)

        with col_s:
            _render_cost_col("This Session", sess, "🔵")
        with col_24:
            _render_cost_col("Past 24 Hours", h24, "🕐")
        with col_tot:
            _render_cost_col("All Time", total, "📊")

        with col_live:
            st.markdown(
                "<div style='text-align:center;padding:6px 0 2px 0'>"
                "<span style='font-size:1.4rem'>⚡</span>&nbsp;"
                "<strong style='font-size:.95rem'>Last Analysis</strong></div>",
                unsafe_allow_html=True,
            )
            if last_run is not None:
                color = _cost_color(last_run)
                st.markdown(
                    f"<div style='text-align:center;margin-bottom:4px'>"
                    f"<span style='font-size:1.8rem;font-weight:700;color:{color}'>"
                    f"${last_run:.4f}</span></div>",
                    unsafe_allow_html=True,
                )
                if last_tokens:
                    st.caption(
                        f"**{_fmt_tokens(last_tokens.get('total', 0))}** tokens  \n"
                        f"↑ {_fmt_tokens(last_tokens.get('input', 0))} in · "
                        f"↓ {_fmt_tokens(last_tokens.get('output', 0))} out"
                    )
            else:
                st.markdown(
                    "<div style='text-align:center;color:#888;padding:12px 0'>"
                    "Run an analysis to see cost</div>",
                    unsafe_allow_html=True,
                )

        st.caption(
            "Anthropic — Agents: claude-sonnet-4-6 · $3/M in · $15/M out  "
            "· Synthesis: claude-opus-4-6 · $15/M in · $75/M out  "
            "· Groq (llama-3.3-70b) = $0.00  · Ollama = $0.00  "
            "· Prompt caching active on Anthropic (system prompts cached → 90% cheaper on repeat runs)"
        )

        # ── Predicted cost per ticker analysis (by model) ────────────
        st.divider()
        st.markdown("##### 🔮 Estimated Cost per Ticker Analysis")

        est = _cost_tracker.get_per_analysis_estimate()
        avg_in = est["input"]
        avg_out = est["output"]
        avg_cw = est["cache_write"]
        avg_cr = est["cache_read"]

        cost_opus   = predict_analysis_cost(
            "claude-opus-4-6", avg_in, avg_out, avg_cw, avg_cr
        )
        cost_sonnet = predict_analysis_cost(
            "claude-sonnet-4-6", avg_in, avg_out, avg_cw, avg_cr
        )
        cost_haiku  = predict_analysis_cost(
            "claude-haiku-4-5-20251001", avg_in, avg_out, avg_cw, avg_cr
        )

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric(
            "Current mix",
            f"${est['cost']:.4f}" if est["analyses"] > 0 else "—",
            help="Historical avg (3 Sonnet agents + 1 Opus synthesis + 1 Sonnet parse)",
        )
        p2.metric(
            "All Opus 4.6", f"${cost_opus:.4f}",
            help="$15 in / $75 out per 1M tokens",
        )
        p3.metric(
            "All Sonnet 4.6", f"${cost_sonnet:.4f}",
            help="$3 in / $15 out per 1M tokens",
        )
        p4.metric(
            "All Haiku 4.5", f"${cost_haiku:.4f}",
            help="$0.80 in / $4 out per 1M tokens",
        )
        p5.metric(
            "Groq / Ollama", "$0.0000",
            help="Free — no per-token charge",
        )

        _avg_total = avg_in + avg_out + avg_cw + avg_cr
        if est["analyses"] > 0:
            st.caption(
                f"Based on **{est['analyses']}** past Anthropic analyses "
                f"(~{_avg_total:,} tokens/run). Estimates assume each model would "
                "use the same prompt sizes — actual output length may vary."
            )
        else:
            st.caption(
                f"Showing default estimates — no past Anthropic runs found "
                f"(assumes ~{_avg_total:,} tokens/analysis). "
                "Run one Anthropic analysis to calibrate to your prompts."
            )


_cost_tracker_panel()
st.divider()

# ── Input row ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    default_ticker = st.session_state.get("selected_ticker", "")
    ticker_raw = st.text_input(
        "**Stock Ticker**",
        value=default_ticker,
        placeholder="e.g. AAPL, MSFT, DRS, NVDA",
        help="Enter any US stock symbol listed on Yahoo Finance",
    )
    ticker = ticker_raw.upper().strip()

with col2:
    provider_choice = st.radio(
        "**LLM Provider**",
        ["🟡 Groq API (free)", "🔵 Anthropic API (paid)", "🟢 Local Ollama (free)"],
        horizontal=True,
        help=(
            "Groq: FREE, built-in, no key needed  |  "
            "Anthropic: highest quality, requires your own API key  |  "
            "Ollama: free local, requires Ollama installed and running"
        ),
    )
    if provider_choice.startswith("🟡"):
        provider = "groq"
    elif provider_choice.startswith("🔵"):
        provider = "api"
    else:
        provider = "ollama"

with col3:
    if provider == "ollama":
        from llm_client import get_available_ollama_models
        available_models = get_available_ollama_models()
        if available_models:
            ollama_model = st.selectbox(
                "**Ollama Model**",
                options=available_models,
                help="Models currently pulled in Ollama. To add more: ollama pull <model>",
            )
        else:
            st.warning("Ollama not running or no models pulled.", icon="⚠️")
            ollama_model = st.text_input(
                "**Ollama Model**",
                value="llama3.2",
                help="Start Ollama and pull a model: ollama pull llama3.2",
            )
    elif provider == "groq":
        ollama_model = st.selectbox(
            "**Groq Model**",
            options=[
                "llama-3.3-70b-versatile",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "llama-3.1-8b-instant",
                "qwen/qwen3-32b",
            ],
            help=(
                "llama-3.3-70b-versatile = best quality (recommended)  ·  "
                "llama-4-maverick = latest Llama 4  ·  "
                "llama-3.1-8b-instant = fastest, lowest quality"
            ),
        )
    else:
        ollama_model = "llama3.2"

# ── Action buttons ─────────────────────────────────────────────────
# Detect saved reports for the current ticker so we can show the View button.
saved_reports = get_saved_reports(ticker) if ticker else []
latest_date, latest_path = (saved_reports[0] if saved_reports else (None, None))

col_run, col_view, col_fav, col_spacer = st.columns([1, 1, 1, 1])

with col_run:
    run_clicked = st.button(
        "🚀 Run Analysis",
        type="primary",
        disabled=not ticker,
        use_container_width=True,
    )

with col_view:
    view_clicked = False
    if ticker and latest_date:
        view_clicked = st.button(
            f"📂 View {latest_date}",
            use_container_width=True,
            help=f"Display the saved analysis from {latest_date} without re-running",
        )

with col_fav:
    if ticker:
        favorites = load_favorites()
        already_saved = ticker in favorites
        label = "✅ In Favorites" if already_saved else "⭐ Add to Favorites"
        if st.button(label, disabled=already_saved, use_container_width=True):
            favorites.append(ticker)
            save_favorites(favorites)
            st.success(f"{ticker} added to favorites!")
            st.rerun()

# ── If the ticker changed, clear any pinned cache view ────────────
pinned = st.session_state.get("view_cache")
if pinned and pinned.get("ticker") != ticker:
    st.session_state.pop("view_cache", None)
    pinned = None

# ── "View latest" button pressed — pin it in session state ────────
if view_clicked and latest_path:
    st.session_state["view_cache"] = {
        "ticker": ticker,
        "date": latest_date,
        "html_path": latest_path,
    }
    pinned = st.session_state["view_cache"]

# ── Show pinned cached report (persists across reruns) ────────────
if pinned and not run_clicked:
    st.info(
        f"📂 Showing saved analysis for **{pinned['ticker']}** "
        f"from **{pinned['date']}**. "
        f"Click **🚀 Run Analysis** to generate a fresh report."
    )
    # List all saved dates for this ticker as a quick selector
    if len(saved_reports) > 1:
        date_options = [d for d, _ in saved_reports]
        chosen_date = st.selectbox(
            "Switch to another saved analysis:",
            options=date_options,
            index=date_options.index(pinned["date"]) if pinned["date"] in date_options else 0,
            key="date_selector",
        )
        if chosen_date != pinned["date"]:
            chosen_path = next(p for d, p in saved_reports if d == chosen_date)
            st.session_state["view_cache"] = {
                "ticker": ticker,
                "date": chosen_date,
                "html_path": chosen_path,
            }
            st.rerun()

    _render_html_report(pinned["html_path"])
    st.stop()

# ── Scan — results display ────────────────────────────────────────
# A scan triggers when: a single ticker button was just pressed, scan-all
# was pressed, or there are already active scanning tickers to show results for.
_scan_single = st.session_state.pop("scan_single", None)
_scan_all = st.session_state.pop("scan_all", False)
_active_scans = st.session_state.get("scanning_tickers", set())

# Determine which tickers to scan right now
_tickers_to_scan = set()
if _scan_single:
    _tickers_to_scan.add(_scan_single)
if _scan_all:
    _tickers_to_scan = set(_active_scans)
elif _active_scans and not _scan_single:
    # Show persisted results for all active scans on every rerun
    _tickers_to_scan = set(_active_scans)

if _tickers_to_scan:
    from scanner import scan_ticker, is_trading_hours, create_demo_analysis
    from alerts.telegram import is_configured as telegram_configured, send_zone_alert
    from analysis_store import get_analysis_age_days

    # Check trading hours
    is_open, market_status = is_trading_hours()
    if not is_open:
        st.warning(f"Market is closed: {market_status}. Prices may be stale.")
    else:
        st.info(f"Market status: {market_status}")

    # Ensure DEMO analysis exists
    if "DEMO" in _tickers_to_scan and get_analysis_age_days("DEMO") is None:
        create_demo_analysis()

    # Run scan for each active ticker
    is_fresh_scan = bool(_scan_single or _scan_all)
    if is_fresh_scan:
        scan_label = (
            f"Scanning {_scan_single}..." if _scan_single
            else f"Scanning {len(_tickers_to_scan)} tickers..."
        )
        with st.status(scan_label, expanded=True) as scan_status:
            results = [scan_ticker(t, auto_refresh=True) for t in sorted(_tickers_to_scan)]
            scan_status.update(label="Scan complete", state="complete")
        # Cache results so they persist across reruns without re-fetching
        st.session_state["scan_results"] = {r.ticker: r for r in results}
    else:
        # Re-display cached results for active tickers
        cached = st.session_state.get("scan_results", {})
        results = [cached[t] for t in sorted(_tickers_to_scan) if t in cached]
        # Scan any tickers that don't have cached results yet
        missing = sorted(_tickers_to_scan - set(cached.keys()))
        if missing:
            with st.status(f"Scanning {', '.join(missing)}...", expanded=True) as scan_status:
                for t in missing:
                    if t == "DEMO" and get_analysis_age_days("DEMO") is None:
                        create_demo_analysis()
                    r = scan_ticker(t, auto_refresh=True)
                    results.append(r)
                    st.session_state.setdefault("scan_results", {})[t] = r
                scan_status.update(label="Scan complete", state="complete")

    if not results:
        st.info("No scan results. Click 🔍 on a ticker to start scanning.")
    else:
        # ── Results display ───────────────────────────────────────
        st.markdown(f"### 🔍 Scanner — {len(results)} ticker(s)")

        alerts_triggered = []

        for r in results:
            if r.error:
                st.warning(f"**{r.ticker}**: {r.error}")
                continue

            # Zone status badge
            if r.below_support:
                status_icon, status_text = "🔴", "BELOW SUPPORT"
            elif r.in_buy_zone:
                status_icon, status_text = "🟢", "IN BUY ZONE"
            elif r.above_resistance:
                status_icon, status_text = "🟣", "ABOVE RESISTANCE"
            elif r.in_sell_zone:
                status_icon, status_text = "🟡", "IN SELL ZONE"
            else:
                status_icon, status_text = "⚪", "Between zones"

            col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])

            col_a.metric(
                r.ticker,
                f"${r.current_price:.2f}",
                delta=f"{status_icon} {status_text}",
                delta_color="off",
            )

            if r.primary_support:
                col_b.metric(
                    "Buy Zone (Support)",
                    f"${r.primary_support:.2f}",
                    delta=f"{r.distance_to_support_pct:+.1f}%",
                    delta_color="inverse",
                )
            else:
                col_b.metric("Buy Zone", "N/A")

            if r.primary_resistance:
                col_c.metric(
                    "Sell Zone (Resistance)",
                    f"${r.primary_resistance:.2f}",
                    delta=f"{r.distance_to_resistance_pct:+.1f}%",
                    delta_color="normal",
                )
            else:
                col_c.metric("Sell Zone", "N/A")

            col_d.metric("Verdict", r.verdict or "N/A")

            details = []
            if r.composite_score:
                details.append(f"Score: **{r.composite_score:.1f}/10**")
            if r.price_target:
                details.append(f"Target: **{r.price_target}**")
            if r.analysis_refreshed:
                details.append("🔄 *Analysis refreshed*")
            if details:
                st.caption(" · ".join(details))

            if r.has_alert:
                alerts_triggered.append(r)

            st.divider()

        # ── Alert summary + Telegram ──────────────────────────────
        if alerts_triggered:
            buy_hits = [r for r in alerts_triggered if r.in_buy_zone or r.below_support]
            sell_hits = [r for r in alerts_triggered if r.in_sell_zone or r.above_resistance]

            if buy_hits:
                st.success(f"**BUY ZONE: {', '.join(r.ticker for r in buy_hits)}**")
            if sell_hits:
                st.error(f"**SELL ZONE: {', '.join(r.ticker for r in sell_hits)}**")

            # Send Telegram alerts only when price FIRST enters the zone (state change).
            # Uses persisted alert_state.json — avoids repeated messages on every scan.
            if is_fresh_scan:
                if telegram_configured():
                    from scanner import check_and_update_zone_state
                    # Update state for all scanned tickers (tracks zone exits too)
                    for r in results:
                        if r.error:
                            continue
                        if r.has_alert:
                            if check_and_update_zone_state(r.ticker, r.zone_status):
                                tg_result = send_zone_alert(r)
                                if tg_result["success"]:
                                    st.success(f"✅ Telegram alert sent for **{r.ticker}** ({r.zone_status})")
                                else:
                                    st.warning(f"Telegram failed for {r.ticker}: {tg_result['error']}")
                            else:
                                st.caption(f"📵 {r.ticker}: already in {r.zone_status} — no repeat alert")
                        else:
                            # Price left the zone → reset state so next entry fires
                            check_and_update_zone_state(r.ticker, r.zone_status)
                else:
                    st.info(
                        "📱 **Telegram alerts not configured.** To receive alerts:\n\n"
                        "1. Message **@BotFather** on Telegram → `/newbot`\n"
                        "2. Copy the bot token\n"
                        "3. Start a chat with your bot, send any message\n"
                        "4. Get your chat\\_id: `https://api.telegram.org/bot<TOKEN>/getUpdates`\n"
                        "5. Add to `.env`:\n\n"
                        "`TELEGRAM_BOT_TOKEN=your_bot_token`\n"
                        "`TELEGRAM_CHAT_ID=your_chat_id`"
                    )
        elif is_fresh_scan:
            st.info("No tickers in buy or sell zone right now.")

# ── Run analysis ───────────────────────────────────────────────────
if run_clicked and ticker:
    # Pre-flight: make sure the chosen provider has a key available.
    _missing_key = (
        (provider == "api"  and not os.environ.get("ANTHROPIC_API_KEY"))
        or (provider == "groq" and not os.environ.get("GROQ_API_KEY"))
    )
    if _missing_key:
        _name = "Anthropic" if provider == "api" else "Groq"
        st.error(
            f"No {_name} API key set. Open the **🔑 API Keys** panel in the "
            f"sidebar and paste your key, or pick a different provider."
        )
        st.stop()

    # Clear any pinned cache so the new result is shown cleanly.
    st.session_state.pop("view_cache", None)

    with st.status(f"Analyzing **{ticker}**...", expanded=True) as status:
        st.write("🔄 Running 3 agents in parallel (technical · fundamental · sentiment)...")
        _pre_stats = _cost_tracker.get_session_stats()
        try:
            report, html_path = run_analysis_cached(ticker, provider, ollama_model)
            # Capture per-run cost delta (only for paid Anthropic calls)
            _post_stats = _cost_tracker.get_session_stats()
            _run_cost = round(_post_stats["cost"] - _pre_stats["cost"], 4)
            _run_tokens = {
                k: _post_stats[k] - _pre_stats[k]
                for k in ("input", "output", "cache_write", "cache_read", "total")
            }
            st.session_state["last_run_cost"] = _run_cost
            st.session_state["last_run_tokens"] = _run_tokens
            # Invalidate the 24h/total cache so it refreshes
            st.session_state.pop("cost_24h_cache", None)
            status.update(label=f"✅ Analysis complete for **{ticker}**", state="complete")
        except ConnectionError as e:
            status.update(label="❌ Connection error", state="error")
            st.error(str(e))
            st.stop()
        except ValueError as e:
            msg = str(e)
            if "ollama pull" in msg.lower() or "not pulled" in msg.lower():
                status.update(label="❌ Ollama model not available", state="error")
                st.error(msg)
                st.code("ollama list", language="bash")
            elif "groq_api_key" in msg.lower() or "console.groq.com" in msg.lower():
                status.update(label="❌ Groq API key not configured", state="error")
                st.error(msg)
                st.info("Add your free Groq API key to `.env`:  `GROQ_API_KEY=your-key-here`")
            else:
                status.update(label="❌ Invalid ticker or data error", state="error")
                st.error(msg)
                st.info(f"Make sure '{ticker}' is a valid Yahoo Finance ticker.")
            st.stop()
        except Exception as e:
            status.update(label="❌ Unexpected error", state="error")
            st.error(f"Unexpected error: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # ── Result summary banner ──────────────────────────────────────
    verdict_colors = {
        "STRONG BUY": "🟢",
        "BUY": "🟩",
        "HOLD": "🟡",
        "SELL": "🟠",
        "STRONG SELL": "🔴",
    }
    icon = verdict_colors.get(report.verdict, "⚪")
    st.success(
        f"{icon} **{report.verdict}** &nbsp;|&nbsp; "
        f"Score: **{report.composite_score:.1f}/10** &nbsp;|&nbsp; "
        f"Confidence: **{report.confidence_pct:.0f}%** &nbsp;|&nbsp; "
        f"Horizon: **{report.time_horizon}**"
    )

    # ── Embedded HTML report ───────────────────────────────────────
    _render_html_report(html_path)
