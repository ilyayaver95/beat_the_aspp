"""
pages/3_Portfolio_Baseline.py
=============================
Portfolio Baseline tracker — a sibling of the Portfolio Tracker page.

Multi-tenant: every row is scoped by user_id, persisted in the shared
external DB (Postgres on Cloud, SQLite locally).

Workflow:
  1. Enter the CURRENT state of your portfolio as a baseline:
       - One row per ticker: current value $ and total return %
       - Free cash sitting on the side
       - An "as-of" date (defaults to today)
  2. From that baseline forward, add BUY / SELL trades as you make them.
  3. The dashboard rolls baseline + trades through live prices to show
     value, unrealized P&L, realized P&L, and total return %.
"""

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime

from auth import require_login, render_sidebar_user_box
from portfolio_baseline_db import (
    init_db, get_meta, set_meta,
    get_positions,
    replace_all_positions_from_value_return,
    qty_cost_to_value_return,
    add_trade, update_trade, delete_trade,
    get_all_trades, get_trades_for_ticker,
    compute_combined_position, enrich_with_price,
    add_transfer, update_transfer, delete_transfer,
    get_all_transfers, get_transfer_totals,
)
from trade_chart import build_trade_picture

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Baseline",
    page_icon="🧭",
    layout="wide",
)

# ── Authentication gate ──────────────────────────────────────────────────
_USER = require_login()
_USER_ID = int(_USER["id"])
render_sidebar_user_box()

init_db()


# ── Price helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    prices = {}
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).fast_info
            price = (getattr(info, "last_price", None)
                     or getattr(info, "previous_close", None))
            prices[tkr] = float(price) if price else 0.0
        except Exception:
            prices[tkr] = 0.0
    return prices


# ── Formatting helpers ─────────────────────────────────────────────────────

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


def _allocation_pie(labels: list[str], values: list[float], title: str) -> go.Figure:
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
        marker=dict(line=dict(color="#1a1a2e", width=2)),
    )])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=14, color="#eee")),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5,
                    font=dict(color="#ddd", size=11)),
        margin=dict(t=40, b=10, l=10, r=10),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eee"),
    )
    return fig


# ── Snapshot builder ───────────────────────────────────────────────────────

def _build_snapshot(user_id: int) -> tuple[list[dict], dict, dict]:
    meta = get_meta(user_id)
    transfers = get_transfer_totals(user_id)
    baseline = {p["ticker"]: p for p in get_positions(user_id)}
    trade_tickers = {t["ticker"] for t in get_all_trades(user_id)}
    all_tickers = sorted(set(baseline.keys()) | trade_tickers)

    baseline_cost_basis = sum(
        b["quantity"] * b["avg_cost_per_share"] for b in baseline.values()
    )
    total_invested = round(
        baseline_cost_basis + meta["free_cash"] + transfers["net"], 2
    )

    if not all_tickers:
        free_cash_now = meta["free_cash"] + transfers["net"]
        total_value   = free_cash_now
        total_gain    = total_value - total_invested
        return [], {
            "total_value":     round(total_value, 2),
            "total_invested":  total_invested,
            "total_gain":      round(total_gain, 2),
            "total_return_pct": round(
                (total_gain / total_invested * 100) if total_invested > 0 else 0.0, 2
            ),
            "total_cost":      0.0,
            "total_unreal":    0.0,
            "total_real":      0.0,
            "free_cash":       round(free_cash_now, 2),
            "open_positions":  0,
            "transfers_in":    transfers["total_in"],
            "transfers_out":   transfers["total_out"],
        }, meta

    prices = _fetch_prices(tuple(all_tickers))

    positions = []
    total_value  = 0.0
    total_cost   = 0.0
    total_unreal = 0.0
    total_real   = 0.0

    for tkr in all_tickers:
        b = baseline.get(tkr)
        b_qty  = b["quantity"]            if b else 0.0
        b_avg  = b["avg_cost_per_share"]  if b else 0.0
        trades = get_trades_for_ticker(user_id, tkr)

        pos = compute_combined_position(b_qty, b_avg, trades)
        pos["ticker"] = tkr
        pos["baseline_qty"]      = round(b_qty, 6)
        pos["baseline_avg_cost"] = round(b_avg, 4)
        pos = enrich_with_price(pos, prices.get(tkr, 0.0))
        positions.append(pos)

        if pos["current_qty"] > 0:
            total_value  += pos["current_value"]
            total_cost   += pos["cost_of_open_position"]
            total_unreal += pos["unrealized_pnl"]
        total_real += pos["realized_pnl"]

    free_cash_now = (
        meta["free_cash"]
        + transfers["net"]
        + sum(p["cash_flow"] for p in positions)
    )
    total_value += free_cash_now

    total_gain        = total_value - total_invested
    total_return_pct  = (total_gain / total_invested * 100) if total_invested > 0 else 0.0

    invested_base = total_cost
    total_pnl     = total_unreal + total_real

    summary = {
        "total_value":      round(total_value, 2),
        "total_invested":   total_invested,
        "total_gain":       round(total_gain, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_cost":       round(invested_base, 2),
        "total_unreal":     round(total_unreal, 2),
        "total_real":       round(total_real, 2),
        "total_pnl":        round(total_pnl, 2),
        "free_cash":        round(free_cash_now, 2),
        "open_positions":   sum(1 for p in positions if p["current_qty"] > 0),
        "transfers_in":     transfers["total_in"],
        "transfers_out":    transfers["total_out"],
    }
    return positions, summary, meta


# ══════════════════════════════════════════════════════════════════
# ── Live dashboard (auto-refreshes every 30 s) ───────────────────
# ══════════════════════════════════════════════════════════════════

@st.fragment(run_every=30)
def _dashboard(user_id: int) -> None:
    positions, summary, meta = _build_snapshot(user_id)

    if (not positions
            and summary["free_cash"] == 0
            and summary["total_invested"] == 0):
        st.info("Set your baseline below to start tracking.")
        return

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Portfolio Value",  _fmt_currency(summary["total_value"]))
    r1c2.metric("Total Invested",   _fmt_currency(summary["total_invested"]))
    r1c3.metric(
        "Total Gain",
        _fmt_currency(summary["total_gain"]),
        delta=_fmt_pct(summary["total_return_pct"]),
        delta_color="normal",
    )
    r1c4.metric("Total Return",     _fmt_pct(summary["total_return_pct"]))

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Free Cash",      _fmt_currency(summary["free_cash"]))
    r2c2.metric("Unrealized P&L", _fmt_currency(summary["total_unreal"]))
    r2c3.metric("Realized P&L",   _fmt_currency(summary["total_real"]))
    r2c4.metric("Open Positions", str(summary["open_positions"]))

    caption_parts = []
    if meta["as_of_date"]:
        caption_parts.append(f"Baseline set as of **{meta['as_of_date']}**")
    if summary["transfers_in"] or summary["transfers_out"]:
        caption_parts.append(
            f"Transfers in: {_fmt_currency(summary['transfers_in'])} · "
            f"out: {_fmt_currency(summary['transfers_out'])}"
        )
    if caption_parts:
        st.caption("  ·  ".join(caption_parts))

    st.divider()

    open_pos   = [p for p in positions if p["current_qty"] > 0]
    closed_pos = [p for p in positions if p["current_qty"] == 0]

    if open_pos:
        st.markdown("#### Open Positions")
        cols = st.columns(min(len(open_pos), 4))
        for i, pos in enumerate(open_pos):
            col   = cols[i % 4]
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

    if open_pos or summary["free_cash"] > 0:
        st.markdown("#### Portfolio Allocation")
        stock_labels = [p["ticker"] for p in open_pos]
        stock_values = [p["current_value"] for p in open_pos]
        free_cash = max(0.0, summary["free_cash"])

        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            if stock_labels:
                st.plotly_chart(
                    _allocation_pie(stock_labels, stock_values, "Stocks only"),
                    use_container_width=True,
                    key="alloc_pie_p3_stocks",
                )
            else:
                st.info("No open stock positions to plot.")
        with col_pie2:
            if stock_labels or free_cash > 0:
                all_labels = stock_labels + (["Cash"] if free_cash > 0 else [])
                all_values = stock_values + ([free_cash] if free_cash > 0 else [])
                st.plotly_chart(
                    _allocation_pie(all_labels, all_values, "Including cash"),
                    use_container_width=True,
                    key="alloc_pie_p3_with_cash",
                )

    st.markdown("#### Position Details")
    rows = []
    for pos in positions:
        rows.append({
            "Ticker":         pos["ticker"],
            "Baseline Qty":   pos["baseline_qty"],
            "Baseline Avg":   pos["baseline_avg_cost"],
            "Shares Now":     pos["current_qty"],
            "Avg Cost":       pos["avg_cost_basis"],
            "Curr Price":     pos["current_price"],
            "Value ($)":      pos["current_value"],
            "Unrealized P&L": pos["unrealized_pnl"],
            "Realized P&L":   pos["realized_pnl"],
            "Total P&L":      pos["total_pnl"],
            "P&L %":          pos["total_pnl_pct"],
            "Status":         "Open" if pos["current_qty"] > 0 else "Closed",
        })
    df = pd.DataFrame(rows)

    def _color_pnl(val):
        if isinstance(val, (int, float)):
            color = "#2ecc71" if val > 0 else ("#e74c3c" if val < 0 else "")
            return f"color: {color}" if color else ""
        return ""

    styled = (
        df.style
        .applymap(_color_pnl,
                  subset=["Unrealized P&L", "Realized P&L", "Total P&L", "P&L %"])
        .format({
            "Baseline Qty":   "{:.4g}",
            "Baseline Avg":   "${:,.2f}",
            "Shares Now":     "{:.4g}",
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
# ── Page layout ───────────────────────────────────────────════════
# ══════════════════════════════════════════════════════════════════

st.title("🧭 Portfolio Baseline")
st.caption(
    "Snapshot your starting portfolio, then track every BUY / SELL "
    "from that point onward."
)
st.divider()

_dashboard(_USER_ID)

st.divider()

# ── Trade Picture ──────────────────────────────────────────────────
st.markdown("### 📊 Trade Picture")
st.caption(
    "Pick a ticker to see its price chart with your BUY (▲ green) and "
    "SELL (▼ red) points overlaid. Baseline holdings are shown as a "
    "synthetic BUY on the baseline date."
)

_baseline_positions = {p["ticker"]: p for p in get_positions(_USER_ID)}
_post_baseline_trades = get_all_trades(_USER_ID)
_picker_meta = get_meta(_USER_ID)

_picker_tickers: list[str] = []
_seen_pick = set()
# Prefer most recently sold first.
for t in sorted(
    (t for t in _post_baseline_trades if (t.get("action") or "").upper() == "SELL"),
    key=lambda t: t.get("trade_date", ""), reverse=True,
):
    tkr = (t.get("ticker") or "").upper()
    if tkr and tkr not in _seen_pick:
        _seen_pick.add(tkr); _picker_tickers.append(tkr)
for t in _post_baseline_trades:
    tkr = (t.get("ticker") or "").upper()
    if tkr and tkr not in _seen_pick:
        _seen_pick.add(tkr); _picker_tickers.append(tkr)
for tkr in _baseline_positions:
    if tkr not in _seen_pick:
        _seen_pick.add(tkr); _picker_tickers.append(tkr)

if not _picker_tickers:
    st.info("No trades or baseline positions yet — your trade picture will appear once you add some.")
else:
    pic_ticker = st.selectbox(
        "Ticker", _picker_tickers, key="trade_pic_ticker_p3",
    )

    if pic_ticker:
        ticker_trades = list(get_trades_for_ticker(_USER_ID, pic_ticker))
        # If we have a baseline holding, prepend a synthetic BUY on the
        # baseline date so the chart shows the starting point.
        if pic_ticker in _baseline_positions:
            b = _baseline_positions[pic_ticker]
            if b["quantity"] > 0 and b["avg_cost_per_share"] > 0:
                ticker_trades = [{
                    "action":          "BUY",
                    "trade_date":      _picker_meta.get("as_of_date") or "",
                    "quantity":        b["quantity"],
                    "price_per_share": b["avg_cost_per_share"],
                    "notes":           "baseline",
                }] + ticker_trades

        with st.spinner(f"Loading {pic_ticker} history…"):
            fig, pic_summary = build_trade_picture(pic_ticker, ticker_trades)

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Bought (qty)", f"{pic_summary['total_bought_qty']:.4g}")
        s_col2.metric("Sold (qty)",   f"{pic_summary['total_sold_qty']:.4g}")
        s_col3.metric("Avg buy",  _fmt_currency(pic_summary["avg_buy_price"]))
        s_col4.metric(
            "Avg sell",
            _fmt_currency(pic_summary["avg_sell_price"]),
            delta=(_fmt_pct(pic_summary["return_pct"])
                   if pic_summary["total_sold_qty"] > 0 else None),
            delta_color="normal",
        )

        if fig is None:
            st.warning(
                f"Couldn't load price history for **{pic_ticker}**. "
                "Yahoo may be rate-limiting — try again in a minute."
            )
        else:
            st.plotly_chart(
                fig, use_container_width=True,
                key=f"trade_pic_fig_p3_{pic_ticker}",
            )

st.divider()

# ── Baseline setup section ─────────────────────────────────────────
_existing_meta = get_meta(_USER_ID)
_existing_positions = get_positions(_USER_ID)
_baseline_is_empty = not _existing_positions and _existing_meta["free_cash"] == 0

with st.expander(
    "🧭 Baseline Setup — initial portfolio state",
    expanded=_baseline_is_empty,
):
    st.caption(
        "Enter what you hold **right now**. The dashboard treats this as the "
        "starting point — every trade you add below is rolled forward from here."
    )

    col_cash, col_date = st.columns(2)
    with col_cash:
        free_cash_in = st.number_input(
            "Free cash (unused $)",
            min_value=0.0,
            step=100.0,
            value=float(_existing_meta["free_cash"]),
            format="%.2f",
        )
    with col_date:
        try:
            _default_date = (
                datetime.strptime(_existing_meta["as_of_date"], "%Y-%m-%d").date()
                if _existing_meta["as_of_date"] else date.today()
            )
        except ValueError:
            _default_date = date.today()
        as_of_in = st.date_input("Baseline date", value=_default_date)

    st.markdown("**Positions held at baseline**")
    st.caption(
        "One row per ticker. Enter the **current $ value** of the position "
        "and your **total return %** so far (positive = up, negative = down). "
        "Use the **+** button at the bottom of the table to add a row."
    )

    _existing_tickers = tuple(p["ticker"] for p in _existing_positions)
    _existing_prices = (
        _fetch_prices(_existing_tickers) if _existing_tickers else {}
    )
    if _existing_positions:
        baseline_rows = []
        for p in _existing_positions:
            price = _existing_prices.get(p["ticker"], 0.0)
            if price > 0:
                value, ret_pct = qty_cost_to_value_return(
                    p["quantity"], p["avg_cost_per_share"], price
                )
            else:
                value, ret_pct = p["quantity"] * p["avg_cost_per_share"], 0.0
            baseline_rows.append({
                "Ticker":           p["ticker"],
                "Value ($)":        round(value, 2),
                "Total Return (%)": round(ret_pct, 2),
            })
    else:
        baseline_rows = [{"Ticker": "", "Value ($)": 0.0, "Total Return (%)": 0.0}]

    baseline_df = pd.DataFrame(baseline_rows)

    edited_baseline = st.data_editor(
        baseline_df,
        key="baseline_editor",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Ticker":   st.column_config.TextColumn("Ticker", max_chars=10),
            "Value ($)": st.column_config.NumberColumn(
                "Value ($)", format="$%.2f", min_value=0.0,
                help="Current dollar value of this position",
            ),
            "Total Return (%)": st.column_config.NumberColumn(
                "Total Return (%)", format="%+.2f%%", min_value=-99.99,
                help="Your gain/loss on this position so far (e.g. 12.5 for +12.5%)",
            ),
        },
    )

    try:
        preview_rows = [
            r for _, r in edited_baseline.iterrows()
            if str(r.get("Ticker", "")).strip()
               and float(r.get("Value ($)") or 0) > 0
        ]
        baseline_value = sum(float(r["Value ($)"]) for r in preview_rows)
        st.caption(
            f"Baseline positions value: **{_fmt_currency(baseline_value)}**  ·  "
            f"Cash: **{_fmt_currency(free_cash_in)}**  ·  "
            f"Total baseline portfolio: "
            f"**{_fmt_currency(baseline_value + free_cash_in)}**"
        )
    except Exception:
        pass

    if st.button("💾 Save Baseline", type="primary", use_container_width=True):
        try:
            payload_rows = []
            for _, row in edited_baseline.iterrows():
                tkr = str(row.get("Ticker", "")).strip().upper()
                value = float(row.get("Value ($)") or 0)
                if not tkr or value <= 0:
                    continue
                payload_rows.append({
                    "ticker":           tkr,
                    "value":            value,
                    "total_return_pct": float(row.get("Total Return (%)") or 0),
                })

            tickers_needed = tuple(r["ticker"] for r in payload_rows)
            _fetch_prices.clear()
            live_prices = _fetch_prices(tickers_needed) if tickers_needed else {}

            missing = [
                t for t in tickers_needed
                if not live_prices.get(t) or live_prices[t] <= 0
            ]
            if missing:
                st.error(
                    "Could not fetch a live price for: "
                    + ", ".join(missing)
                    + ". Try again or verify the ticker symbol."
                )
            else:
                replace_all_positions_from_value_return(
                    _USER_ID, payload_rows, live_prices,
                )
                set_meta(_USER_ID, float(free_cash_in), str(as_of_in))
                _fetch_prices.clear()
                st.success("Baseline saved.")
                st.rerun()
        except ValueError as e:
            st.error(str(e))


st.divider()

# ── Add trade / transfer + history (post-baseline) ─────────────────
col_form, col_history = st.columns([1, 2], gap="large")

with col_form:
    form_tab_trade, form_tab_transfer = st.tabs(
        ["➕ Add Trade", "💵 Add Transfer"]
    )

    with form_tab_trade:
        st.caption("Records a BUY or SELL made AFTER the baseline date.")
        with st.form("baseline_add_trade_form", clear_on_submit=True):
            ticker_in = st.text_input(
                "Ticker", placeholder="e.g. AAPL", max_chars=10
            ).upper().strip()

            action_in = st.radio("Action", ["BUY", "SELL"], horizontal=True)

            _meta = get_meta(_USER_ID)
            try:
                _trade_default_date = (
                    datetime.strptime(_meta["as_of_date"], "%Y-%m-%d").date()
                    if _meta["as_of_date"] else date.today()
                )
            except ValueError:
                _trade_default_date = date.today()

            date_in = st.date_input("Trade Date", value=date.today(),
                                    min_value=_trade_default_date)

            qty_in = st.number_input(
                "Quantity (shares)", min_value=0.0001, step=1.0,
                format="%.4f", value=1.0,
            )
            price_in = st.number_input(
                "Price per Share ($)", min_value=0.0001, step=0.01,
                format="%.4f", value=1.0,
            )
            st.caption(f"Total trade value: **${qty_in * price_in:,.2f}**")

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
                        user_id=_USER_ID,
                        ticker=ticker_in,
                        action=action_in,
                        trade_date=str(date_in),
                        quantity=qty_in,
                        price_per_share=price_in,
                        notes=notes_in,
                    )
                    _fetch_prices.clear()
                    st.success(
                        f"{action_in} recorded: {qty_in:.4g} × **{ticker_in}** "
                        f"@ ${price_in:.2f} (id #{new_id})"
                    )
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    with form_tab_transfer:
        st.caption(
            "Money moving between your bank and the portfolio. "
            "IN = deposit (raises Total Invested), OUT = withdrawal."
        )
        with st.form("baseline_add_transfer_form", clear_on_submit=True):
            direction_in = st.radio(
                "Direction", ["IN", "OUT"], horizontal=True,
                help="IN: bank → portfolio. OUT: portfolio → bank.",
            )
            transfer_date_in = st.date_input(
                "Transfer Date", value=date.today(),
                key="transfer_date_input",
            )
            amount_in = st.number_input(
                "Amount ($)", min_value=0.01, step=100.0,
                format="%.2f", value=100.0,
            )
            transfer_notes_in = st.text_input(
                "Notes (optional)", max_chars=500,
                key="transfer_notes_input",
            )
            transfer_submitted = st.form_submit_button(
                f"{'💰 Deposit' if direction_in == 'IN' else '🏧 Withdraw'}",
                use_container_width=True,
                type="primary",
            )

        if transfer_submitted:
            try:
                new_id = add_transfer(
                    user_id=_USER_ID,
                    direction=direction_in,
                    transfer_date=str(transfer_date_in),
                    amount=amount_in,
                    notes=transfer_notes_in,
                )
                st.success(
                    f"{direction_in} recorded: "
                    f"**{_fmt_currency(amount_in)}** "
                    f"on {transfer_date_in} (id #{new_id})"
                )
                st.rerun()
            except ValueError as e:
                st.error(str(e))

with col_history:
    hist_tab_trades, hist_tab_transfers = st.tabs(
        ["📋 Trades", "💵 Transfers"]
    )

    with hist_tab_trades:
        all_trades = get_all_trades(_USER_ID)

        if not all_trades:
            st.info("No trades yet. Trades you add will appear here.")
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

            st.data_editor(
                df_display,
                key="baseline_trade_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "ID":      st.column_config.NumberColumn(
                                    "ID", disabled=True, width="small"),
                    "Ticker":  st.column_config.TextColumn("Ticker", max_chars=10),
                    "Action":  st.column_config.SelectboxColumn(
                                    "Action", options=["BUY", "SELL"], required=True
                                ),
                    "Date":    st.column_config.TextColumn("Date", help="YYYY-MM-DD"),
                    "Qty":     st.column_config.NumberColumn(
                                    "Qty", format="%.4g", min_value=0.0001),
                    "Price ($)": st.column_config.NumberColumn(
                                    "Price ($)", format="$%.4f", min_value=0.0001),
                    "Notes":   st.column_config.TextColumn("Notes", max_chars=500),
                },
            )

            editor_state = st.session_state.get("baseline_trade_editor", {})
            has_changes = bool(
                editor_state.get("edited_rows")
                or editor_state.get("deleted_rows")
            )

            if has_changes:
                if st.button("💾 Save Changes", type="primary",
                             use_container_width=True,
                             key="save_trade_changes_btn"):
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
                            update_trade(trade_id, user_id=_USER_ID, **merged)
                        except ValueError as e:
                            errors.append(f"Row {row_idx+1}: {e}")

                    for row_idx in editor_state.get("deleted_rows", []):
                        row_idx = int(row_idx)
                        if row_idx >= len(df_display):
                            continue
                        trade_id = int(df_display.iloc[row_idx]["ID"])
                        try:
                            delete_trade(trade_id, _USER_ID)
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

    with hist_tab_transfers:
        all_transfers = get_all_transfers(_USER_ID)

        if not all_transfers:
            st.info("No transfers yet. Deposits and withdrawals will appear here.")
        else:
            df_xfer = pd.DataFrame(all_transfers)
            xfer_cols = {
                "id":            "ID",
                "direction":     "Direction",
                "transfer_date": "Date",
                "amount":        "Amount ($)",
                "notes":         "Notes",
            }
            df_xfer_display = (
                df_xfer[list(xfer_cols.keys())]
                .rename(columns=xfer_cols)
            )

            st.data_editor(
                df_xfer_display,
                key="baseline_transfer_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "ID":        st.column_config.NumberColumn(
                                    "ID", disabled=True, width="small"),
                    "Direction": st.column_config.SelectboxColumn(
                                    "Direction", options=["IN", "OUT"], required=True
                                ),
                    "Date":      st.column_config.TextColumn(
                                    "Date", help="YYYY-MM-DD"),
                    "Amount ($)":st.column_config.NumberColumn(
                                    "Amount ($)", format="$%.2f", min_value=0.01),
                    "Notes":     st.column_config.TextColumn(
                                    "Notes", max_chars=500),
                },
            )

            xfer_state = st.session_state.get("baseline_transfer_editor", {})
            xfer_has_changes = bool(
                xfer_state.get("edited_rows")
                or xfer_state.get("deleted_rows")
            )

            if xfer_has_changes:
                if st.button("💾 Save Changes", type="primary",
                             use_container_width=True,
                             key="save_transfer_changes_btn"):
                    errors = []

                    for row_idx, changes in xfer_state.get("edited_rows", {}).items():
                        row_idx = int(row_idx)
                        if row_idx >= len(df_xfer_display):
                            continue
                        orig = df_xfer_display.iloc[row_idx]
                        transfer_id = int(orig["ID"])
                        merged = {
                            "direction":     changes.get("Direction",  orig["Direction"]),
                            "transfer_date": changes.get("Date",       orig["Date"]),
                            "amount":        changes.get("Amount ($)", orig["Amount ($)"]),
                            "notes":         changes.get("Notes",      orig["Notes"] or ""),
                        }
                        try:
                            update_transfer(transfer_id, user_id=_USER_ID, **merged)
                        except ValueError as e:
                            errors.append(f"Row {row_idx+1}: {e}")

                    for row_idx in xfer_state.get("deleted_rows", []):
                        row_idx = int(row_idx)
                        if row_idx >= len(df_xfer_display):
                            continue
                        transfer_id = int(df_xfer_display.iloc[row_idx]["ID"])
                        try:
                            delete_transfer(transfer_id, _USER_ID)
                        except Exception as e:
                            errors.append(f"Delete row {row_idx+1}: {e}")

                    if errors:
                        for err in errors:
                            st.error(err)
                    else:
                        st.success("Changes saved.")
                        st.rerun()
            else:
                totals = get_transfer_totals(_USER_ID)
                st.caption(
                    f"In: **{_fmt_currency(totals['total_in'])}**  ·  "
                    f"Out: **{_fmt_currency(totals['total_out'])}**  ·  "
                    f"Net: **{_fmt_currency(totals['net'])}**"
                )
