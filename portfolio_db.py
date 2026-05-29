"""
portfolio_db.py
===============
Trade ledger for the Portfolio Tracker page.

Multi-tenant: every row is scoped by user_id, so a single shared database
(Postgres on Cloud, SQLite locally) holds all users' trades safely.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Table, Column, Integer, String, Text, Float, DateTime, Index, text,
)

from db import get_engine, metadata


trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False),
    Column("ticker", String(10), nullable=False),
    Column("action", String(4), nullable=False),       # BUY / SELL
    Column("trade_date", String(10), nullable=False),  # YYYY-MM-DD
    Column("quantity", Float, nullable=False),
    Column("price_per_share", Float, nullable=False),
    Column("notes", Text, nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("idx_trades_user", "user_id"),
    Index("idx_trades_user_ticker", "user_id", "ticker"),
    Index("idx_trades_user_date", "user_id", "trade_date"),
)


def init_db() -> None:
    metadata.create_all(get_engine(), tables=[trades])


# ── Validation ────────────────────────────────────────────────────

def _validate_trade(ticker: str, action: str, trade_date: str,
                    quantity: float, price_per_share: float) -> tuple[str, str]:
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 10:
        raise ValueError("Ticker must be 1-10 characters.")
    action = action.strip().upper()
    if action not in ("BUY", "SELL"):
        raise ValueError("Action must be BUY or SELL.")
    try:
        datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Trade date must be in YYYY-MM-DD format.")
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        raise ValueError("Quantity must be a positive number.")
    if not isinstance(price_per_share, (int, float)) or price_per_share <= 0:
        raise ValueError("Price per share must be a positive number.")
    return ticker, action


# ── CRUD ──────────────────────────────────────────────────────────

def add_trade(user_id: int, ticker: str, action: str, trade_date: str,
              quantity: float, price_per_share: float, notes: str = "") -> int:
    ticker, action = _validate_trade(ticker, action, trade_date, quantity, price_per_share)
    notes = (notes or "").strip()[:500]
    eng = get_engine()
    sql_pg = text(
        "INSERT INTO trades "
        "  (user_id, ticker, action, trade_date, quantity, price_per_share, notes, created_at) "
        "VALUES (:uid, :ticker, :action, :date, :qty, :price, :notes, :now) "
        "RETURNING id"
    )
    sql_lite = text(
        "INSERT INTO trades "
        "  (user_id, ticker, action, trade_date, quantity, price_per_share, notes, created_at) "
        "VALUES (:uid, :ticker, :action, :date, :qty, :price, :notes, :now)"
    )
    params = {
        "uid": int(user_id), "ticker": ticker, "action": action,
        "date": trade_date, "qty": float(quantity), "price": float(price_per_share),
        "notes": notes, "now": datetime.now(timezone.utc),
    }
    with eng.begin() as conn:
        if eng.dialect.name == "postgresql":
            return int(conn.execute(sql_pg, params).scalar_one())
        return int(conn.execute(sql_lite, params).lastrowid)


def update_trade(trade_id: int, user_id: int, ticker: str, action: str,
                 trade_date: str, quantity: float, price_per_share: float,
                 notes: str = "") -> None:
    ticker, action = _validate_trade(ticker, action, trade_date, quantity, price_per_share)
    notes = (notes or "").strip()[:500]
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE trades SET "
                "  ticker=:ticker, action=:action, trade_date=:date, "
                "  quantity=:qty, price_per_share=:price, notes=:notes "
                "WHERE id=:id AND user_id=:uid"
            ),
            {
                "ticker": ticker, "action": action, "date": trade_date,
                "qty": float(quantity), "price": float(price_per_share),
                "notes": notes, "id": int(trade_id), "uid": int(user_id),
            },
        )


def delete_trade(trade_id: int, user_id: int) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM trades WHERE id=:id AND user_id=:uid"),
            {"id": int(trade_id), "uid": int(user_id)},
        )


def get_all_trades(user_id: int) -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM trades WHERE user_id=:uid "
                "ORDER BY trade_date DESC, id DESC"
            ),
            {"uid": int(user_id)},
        ).all()
    return [dict(r._mapping) for r in rows]


def get_trades_for_ticker(user_id: int, ticker: str) -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM trades WHERE user_id=:uid AND ticker=:t "
                "ORDER BY trade_date ASC, id ASC"
            ),
            {"uid": int(user_id), "t": ticker.strip().upper()},
        ).all()
    return [dict(r._mapping) for r in rows]


def get_tickers(user_id: int) -> list[str]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT ticker FROM trades "
                "WHERE user_id=:uid ORDER BY ticker"
            ),
            {"uid": int(user_id)},
        ).all()
    return [r[0] for r in rows]


# ── P&L (unchanged, no DB access) ─────────────────────────────────

def compute_position(trades_for_ticker: list[dict]) -> dict:
    buys = [t for t in trades_for_ticker if t["action"] == "BUY"]
    sells = [t for t in trades_for_ticker if t["action"] == "SELL"]

    total_buy_qty  = sum(t["quantity"] for t in buys)
    total_buy_cost = sum(t["quantity"] * t["price_per_share"] for t in buys)
    total_sell_qty = sum(t["quantity"] for t in sells)
    total_sell_proceeds = sum(t["quantity"] * t["price_per_share"] for t in sells)

    current_qty = total_buy_qty - total_sell_qty
    avg_cost = total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0.0
    realized_pnl = total_sell_proceeds - (total_sell_qty * avg_cost)

    ticker = trades_for_ticker[0]["ticker"] if trades_for_ticker else ""
    return {
        "ticker":                ticker,
        "total_buy_qty":         total_buy_qty,
        "total_sell_qty":        total_sell_qty,
        "current_qty":           current_qty,
        "avg_cost_basis":        round(avg_cost, 4),
        "total_buy_cost":        round(total_buy_cost, 2),
        "total_sell_proceeds":   round(total_sell_proceeds, 2),
        "cost_of_open_position": round(current_qty * avg_cost, 2),
        "realized_pnl":          round(realized_pnl, 2),
        "current_price":         0.0,
        "current_value":         0.0,
        "unrealized_pnl":        0.0,
        "total_pnl":             round(realized_pnl, 2),
        "total_pnl_pct":         0.0,
    }


def enrich_with_price(pos: dict, current_price: float) -> dict:
    cq = pos["current_qty"]
    avg = pos["avg_cost_basis"]
    unreal = round((current_price - avg) * cq, 2) if cq > 0 else 0.0
    cost_open = pos["cost_of_open_position"]
    total_pnl = round(pos["realized_pnl"] + unreal, 2)
    total_pnl_pct = round((total_pnl / cost_open * 100) if cost_open else 0.0, 2)
    return {
        **pos,
        "current_price":  round(current_price, 4),
        "current_value":  round(current_price * cq, 2),
        "unrealized_pnl": unreal,
        "total_pnl":      total_pnl,
        "total_pnl_pct":  total_pnl_pct,
    }
