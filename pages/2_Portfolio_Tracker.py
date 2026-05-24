"""
pages/2_Portfolio_Tracker.py
============================
Portfolio Tracker — Streamlit multi-page app.

Per-user isolation: each authenticated user gets a private SQLite database
derived from their email address. On Streamlit Community Cloud, enable
"Viewer authentication" in app settings — no extra OAuth setup needed.
"""

import hashlib

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import date, datetime

from portfolio_db import (
    init_db, add_trade, update_trade, delete_trade,
    get_all_trades, get_trades_for_ticker, get_tickers,
    compute_position, enrich_with_price,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="💼",
    layout="wide",
)


# ── User identity ──────────────────────────────────────────────────────────

def _get_user_email() -> str | None:
    """Return the logged-in user's email, or None when running locally."""
    try:
        return st.user.email or None
    except Exception:
        return None


def _portfolio_db_path() -> str:
    email = _get_user_email()
    if email:
        safe = hashlib.sha256(email.lower().encode()).hexdigest()[:20]
        return f"data/portfolio_{safe}.db"
    return "data/portfolio_local.db"


# ── Sidebar: user info ─────────────────────────────────────────────────────

_email = _get_user_email()
if _email:
    st.sidebar.markdown(f"**Signed in as**  \n{_email}")
else:
    st.sidebar.caption("Running locally — data saved to portfolio_local.db")

_DB = _portfolio_db_path()
init_db(_DB)


# ── Price helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Fetch current prices for a tuple of tickers. Cached 30s."""
    prices = {}
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            prices[tkr] = float(price) if price else 0.0
        except Exception:
            prices[tkr] = 0.0
    return prices


# ── P&L color helpers ──────────────────────────────────────────────────────

def _pnl_color(value: float) -> str:
    if value > 0:
        return "#2ecc71"
    if value < 0:
        return "#e74c3c"
    return "#aaaaaa"


def _pnl_icon(value: float) -> str:
    if value > 0:
        return "▲"
    if value < 0:
        return "▼"
    return "—"


def _fmt_currency(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


# ── Build full portfolio snapshot ──────────────────────────────────────────

def _build_portfolio(db_path: str) -> tuple[list[dict], dict]:
    tickers = get_tickers(db_path=db_path)
    if not tickers:
        return [], {}

    prices = _fetch_prices(tuple(tickers))

    positions = []
    total_value = 0.0
    total_cost  = 0.0
    total_unreal = 0.0
    total_real   = 0.0

    for tkr in tickers:
        trades = get_trades_for_ticker(tkr, db_path=db_path)
        if not trades:
            continue
        pos = compute_position(trades)
        pos = enrich_with_price(pos, prices.get(tkr, 0.0))
        positions.append(pos)

        if pos["current_qty"] > 0:
            total_value  += pos["current_value"]
            total_cost   += pos["cost_of_open_position"]
            total_unreal += pos["unrealized_pnl"]
        total_real += pos["realized_pnl"]

    total_pnl     = total_unreal + total_real
    total_pnl_pct = (total_unreal / total_cost * 100) if total_cost else 0.0

    summary = {
        "total_value":    round(total_value, 2),
        "total_cost":     round(total_cost, 2),
        "total_unreal":   round(total_unreal, 2),
        "total_real":     round(total_real, 2),
        "total_pnl":      round(total_pnl, 2),
        "total_pnl_pct":  round(total_pnl_pct, 2),
        "open_positions": sum(1 for p in positions if p["current_qty"] > 0),
    }
    return positions, summary


# ══════════════════════════════════════════════════════════════════
# ── Dashboard fragment (auto-refreshes every 30 s) ────────────────
# ══════════════════════════════════════════════════════════════════

@st.fragment(run_every=30)
def _dashboard(db_path: str) -> None:
    positions, summary = _build_portfolio(db_path)

    if not positions:
        st.info("No trades recorded yet. Add your first trade below.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Portfolio Value",   _fmt_currency(summary["total_value"]))
    col2.metric("Invested (open)",   _fmt_currency(summary["total_cost"]))
    col3.metric(
        "Unrealized P&L",
        _fmt_currency(summary["total_unreal"]),
        delta=_fmt_pct(summary["total_pnl_pct"]),
        delta_color="normal",
    )
    col4.metric("Realized P&L",  _fmt_currency(summary["total_real"]))
    col5.metric("Open Positions", str(summary["open_positions"]))

    st.divider()

    open_pos  = [p for p in positions if p["current_qty"] > 0]
    closed_pos = [p for p in positions if p["current_qty"] == 0]

    if open_pos:
        st.markdown("#### Open Positions")
        cols = st.columns(min(len(open_pos), 4))
        for i, pos in enumerate(open_pos):
            col = cols[i % 4]
            color = _pnl_color(pos["unrealized_pnl"])
            icon  = _pnl_icon(pos["unrealized_pnl"])
            col.markdown(
                f"<div style='"
                f"border:1px solid #333;border-radius:8px;padding:12px;"
                f"margin-bottom:8px;background:#1a1a2e'>"
                f"<div style='font-size:1.2rem;font-weight:700'>{pos['ticker']}</div>"
                f"<div style='font-size:1.5rem;font-weight:700'>"
                f"${pos['current_price']:,.2f}</div>"
                f"<div style='color:#aaa;font-size:.85rem'>"
                f"{pos['current_qty']:.4g} shares · avg ${pos['avg_cost_basis']:.2f}</div>"
                f"<div style='color:{color};font-size:1rem;font-weight:600'>"
                f"{icon} {_fmt_currency(pos['unrealized_pnl'])} "
                f"({_fmt_pct(pos['total_pnl_pct'])})</div>"
                f"<div style='color:#888;font-size:.8rem'>"
                f"Value: {_fmt_currency(pos['current_value'])}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Position Details")
    rows = []
    for pos in positions:
        rows.append({
            "Ticker":        pos["ticker"],
            "Shares":        pos["current_qty"],
            "Avg Cost":      pos["avg_cost_basis"],
            "Curr Price":    pos["current_price"],
            "Value ($)":     pos["current_value"],
            "Unrealized P&L":pos["unrealized_pnl"],
            "Realized P&L":  pos["realized_pnl"],
            "Total P&L":     pos["total_pnl"],
            "P&L %":         pos["total_pnl_pct"],
            "Status":        "Open" if pos["current_qty"] > 0 else "Closed",
        })
    df = pd.DataFrame(rows)

    def _color_pnl(val):
        if isinstance(val, (int, float)):
            color = "#2ecc71" if val > 0 else ("#e74c3c" if val < 0 else "")
            return f"color: {color}" if color else ""
        return ""

    styled = (
        df.style
        .applymap(_color_pnl, subset=["Unrealized P&L", "Realized P&L", "Total P&L", "P&L %"])
        .format({
            "Shares":         "{:.4g}",
            "Avg Cost":       "${:,.2f}",
            "Curr Price":     "${:,.2f}",
            "Value ($)":      "${:,.2f}",
            "Unrealized P&L": "${:,.2f}",
            "Realized P&L":   "${:,.2f}",
            "Total P&L":      "${:,.2f}",
            "P&L %":          "{:+.2f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if closed_pos:
        st.caption(f"**{len(closed_pos)} closed position(s):** "
                   + ", ".join(p["ticker"] for p in closed_pos))

    refresh_ts = datetime.now().strftime("%H:%M:%S")
    st.caption(f"Prices auto-refresh every 30 s · Last updated: {refresh_ts}")


# ══════════════════════════════════════════════════════════════════
# ── Page layout ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

st.title("💼 Portfolio Tracker")
st.caption("Track your stock trades · Real-time P&L · No AI tokens used")
st.divider()

_dashboard(_DB)

st.divider()

col_form, col_history = st.columns([1, 2], gap="large")

# ── ADD TRADE form ─────────────────────────────────────────────────
with col_form:
    st.markdown("### ➕ Add Trade")
    with st.form("add_trade_form", clear_on_submit=True):
        ticker_in = st.text_input(
            "Ticker", placeholder="e.g. AAPL", max_chars=10
        ).upper().strip()

        action_in = st.radio("Action", ["BUY", "SELL"], horizontal=True)

        date_in = st.date_input("Trade Date", value=date.today())

        qty_in = st.number_input(
            "Quantity (shares)", min_value=0.0001, step=1.0,
            format="%.4f", value=1.0
        )
        price_in = st.number_input(
            "Price per Share ($)", min_value=0.0001, step=0.01,
            format="%.4f", value=1.0
        )
        total_preview = qty_in * price_in
        st.caption(f"Total trade value: **${total_preview:,.2f}**")

        notes_in = st.text_input("Notes (optional)", max_chars=500)

        submitted = st.form_submit_button(
            f"{'🟢 Add BUY' if action_in == 'BUY' else '🔴 Add SELL'}",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not ticker_in:
            st.error("Please enter a ticker symbol.")
        else:
            try:
                new_id = add_trade(
                    ticker=ticker_in,
                    action=action_in,
                    trade_date=str(date_in),
                    quantity=qty_in,
                    price_per_share=price_in,
                    notes=notes_in,
                    db_path=_DB,
                )
                _fetch_prices.clear()
                st.success(
                    f"{'BUY' if action_in == 'BUY' else 'SELL'} recorded: "
                    f"{qty_in:.4g} × **{ticker_in}** @ ${price_in:.2f} "
                    f"(id #{new_id})"
                )
                st.rerun()
            except ValueError as e:
                st.error(str(e))

# ── TRANSACTION HISTORY ────────────────────────────────────────────
with col_history:
    st.markdown("### 📋 Transaction History")

    all_trades = get_all_trades(db_path=_DB)

    if not all_trades:
        st.info("No trades yet.")
    else:
        df_trades = pd.DataFrame(all_trades)

        display_cols = {
            "id":              "ID",
            "ticker":          "Ticker",
            "action":          "Action",
            "trade_date":      "Date",
            "quantity":        "Qty",
            "price_per_share": "Price ($)",
            "notes":           "Notes",
        }
        df_display = (
            df_trades[list(display_cols.keys())]
            .rename(columns=display_cols)
        )

        def _style_action(val):
            if val == "BUY":
                return "color: #2ecc71; font-weight:600"
            if val == "SELL":
                return "color: #e74c3c; font-weight:600"
            return ""

        edited = st.data_editor(
            df_display,
            key="trade_editor",
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "ID":       st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "Ticker":   st.column_config.TextColumn("Ticker", max_chars=10),
                "Action":   st.column_config.SelectboxColumn(
                                "Action", options=["BUY", "SELL"], required=True
                            ),
                "Date":     st.column_config.TextColumn("Date", help="YYYY-MM-DD"),
                "Qty":      st.column_config.NumberColumn("Qty", format="%.4g", min_value=0.0001),
                "Price ($)":st.column_config.NumberColumn("Price ($)", format="$%.4f", min_value=0.0001),
                "Notes":    st.column_config.TextColumn("Notes", max_chars=500),
            },
        )

        editor_state = st.session_state.get("trade_editor", {})
        has_changes = bool(
            editor_state.get("edited_rows")
            or editor_state.get("deleted_rows")
        )

        if has_changes:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                errors = []

                for row_idx, changes in editor_state.get("edited_rows", {}).items():
                    row_idx = int(row_idx)
                    if row_idx >= len(df_display):
                        continue
                    orig = df_display.iloc[row_idx]
                    trade_id = int(orig["ID"])
                    merged = {
                        "ticker":          changes.get("Ticker",    orig["Ticker"]),
                        "action":          changes.get("Action",    orig["Action"]),
                        "trade_date":      changes.get("Date",      orig["Date"]),
                        "quantity":        changes.get("Qty",       orig["Qty"]),
                        "price_per_share": changes.get("Price ($)", orig["Price ($)"]),
                        "notes":           changes.get("Notes",     orig["Notes"] or ""),
                    }
                    try:
                        update_trade(trade_id, db_path=_DB, **merged)
                    except ValueError as e:
                        errors.append(f"Row {row_idx+1}: {e}")

                for row_idx in editor_state.get("deleted_rows", []):
                    row_idx = int(row_idx)
                    if row_idx >= len(df_display):
                        continue
                    trade_id = int(df_display.iloc[row_idx]["ID"])
                    try:
                        delete_trade(trade_id, db_path=_DB)
                    except Exception as e:
                        errors.append(f"Delete row {row_idx+1}: {e}")

                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    _fetch_prices.clear()
                    st.success("Changes saved.")
                    st.rerun()
        else:
            st.caption(
                "Edit cells directly to update a trade · "
                "Select a row and press Delete to remove it · "
                "Then click **Save Changes**"
            )
