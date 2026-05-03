"""
portfolio_db.py
===============
SQLite database layer for the Portfolio Tracker.

SECURITY:
  - Every query uses ? parameterized placeholders — no SQL injection possible.
  - All user inputs are validated and sanitized before hitting the DB.
  - WAL journal mode: safe for concurrent reads from Streamlit reruns.

SCHEMA:
  trades — one row per BUY or SELL transaction. P&L is always computed
           on-the-fly from this table; nothing is stored pre-computed
           (avoids stale data).

USAGE:
  from portfolio_db import init_db, add_trade, get_all_trades, ...
  init_db()  # call once at app startup
"""

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "data/portfolio.db"

# ── Schema ─────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    action          TEXT    NOT NULL CHECK(action IN ('BUY','SELL')),
    trade_date      TEXT    NOT NULL,          -- YYYY-MM-DD
    quantity        REAL    NOT NULL CHECK(quantity > 0),
    price_per_share REAL    NOT NULL CHECK(price_per_share > 0),
    notes           TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL           -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_date   ON trades(trade_date);
"""

# ── Internal connection helper ─────────────────────────────────────────────

@contextmanager
def _db():
    """Yield a committed, auto-closed SQLite connection."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Validation helpers ─────────────────────────────────────────────────────

def _validate_trade(ticker: str, action: str, trade_date: str,
                    quantity: float, price_per_share: float) -> tuple[str, str]:
    """
    Validate and normalise trade fields.
    Returns (clean_ticker, clean_action).
    Raises ValueError with a descriptive message on bad input.
    """
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


# ── Public API ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables and indexes if they don't exist. Safe to call on every startup."""
    with _db() as conn:
        conn.executescript(_DDL)


def add_trade(
    ticker: str,
    action: str,
    trade_date: str,
    quantity: float,
    price_per_share: float,
    notes: str = "",
) -> int:
    """
    Insert a new trade record.

    Returns:
        The auto-assigned integer id of the new row.

    Raises:
        ValueError — on invalid inputs (caught by the UI layer)
    """
    ticker, action = _validate_trade(ticker, action, trade_date, quantity, price_per_share)
    notes = notes.strip()[:500]  # cap note length
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (ticker, action, trade_date, quantity, price_per_share, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ticker, action, trade_date,
             float(quantity), float(price_per_share),
             notes,
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def update_trade(
    trade_id: int,
    ticker: str,
    action: str,
    trade_date: str,
    quantity: float,
    price_per_share: float,
    notes: str = "",
) -> None:
    """Update an existing trade by its id."""
    ticker, action = _validate_trade(ticker, action, trade_date, quantity, price_per_share)
    notes = notes.strip()[:500]
    with _db() as conn:
        conn.execute(
            """UPDATE trades
               SET ticker=?, action=?, trade_date=?, quantity=?,
                   price_per_share=?, notes=?
               WHERE id=?""",
            (ticker, action, trade_date,
             float(quantity), float(price_per_share),
             notes, int(trade_id)),
        )


def delete_trade(trade_id: int) -> None:
    """Delete a trade by its id."""
    with _db() as conn:
        conn.execute("DELETE FROM trades WHERE id=?", (int(trade_id),))


def get_all_trades() -> list[dict]:
    """Return all trades, newest first."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY trade_date DESC, id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_trades_for_ticker(ticker: str) -> list[dict]:
    """Return all trades for one ticker, oldest first (for cost basis calc)."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE ticker=? ORDER BY trade_date ASC, id ASC",
            (ticker.strip().upper(),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_tickers() -> list[str]:
    """Return the list of distinct tickers that have at least one trade."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM trades ORDER BY ticker"
        ).fetchall()
    return [r["ticker"] for r in rows]


# ── P&L Calculations ───────────────────────────────────────────────────────

def compute_position(trades: list[dict]) -> dict:
    """
    Compute the full P&L position for a list of trades for ONE ticker.
    Uses the average cost basis method.

    Args:
        trades: All trades for a single ticker, sorted by date ascending.

    Returns dict with:
        ticker, total_buy_qty, total_sell_qty, current_qty,
        avg_cost_basis, total_invested, total_sell_proceeds,
        realized_pnl, unrealized_pnl (0 — caller fills in with live price),
        total_pnl (realized only here; caller adds unrealized)
    """
    buys  = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]

    total_buy_qty  = sum(t["quantity"] for t in buys)
    total_buy_cost = sum(t["quantity"] * t["price_per_share"] for t in buys)
    total_sell_qty = sum(t["quantity"] for t in sells)
    total_sell_proceeds = sum(t["quantity"] * t["price_per_share"] for t in sells)

    current_qty = total_buy_qty - total_sell_qty
    avg_cost = total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0.0

    # Realized P&L = sell proceeds minus what those shares cost us on average
    realized_pnl = total_sell_proceeds - (total_sell_qty * avg_cost)

    ticker = trades[0]["ticker"] if trades else ""
    return {
        "ticker":               ticker,
        "total_buy_qty":        total_buy_qty,
        "total_sell_qty":       total_sell_qty,
        "current_qty":          current_qty,
        "avg_cost_basis":       round(avg_cost, 4),
        "total_buy_cost":       round(total_buy_cost, 2),
        "total_sell_proceeds":  round(total_sell_proceeds, 2),
        "cost_of_open_position": round(current_qty * avg_cost, 2),
        "realized_pnl":         round(realized_pnl, 2),
        # caller fills these in after fetching live price:
        "current_price":        0.0,
        "current_value":        0.0,
        "unrealized_pnl":       0.0,
        "total_pnl":            round(realized_pnl, 2),
        "total_pnl_pct":        0.0,
    }


def enrich_with_price(pos: dict, current_price: float) -> dict:
    """
    Fill in the price-dependent fields of a position dict.
    Call this after compute_position() once you have a live price.
    """
    cq    = pos["current_qty"]
    avg   = pos["avg_cost_basis"]
    unreal = round((current_price - avg) * cq, 2) if cq > 0 else 0.0
    cost_open = pos["cost_of_open_position"]
    total_pnl = round(pos["realized_pnl"] + unreal, 2)
    total_pnl_pct = round((total_pnl / cost_open * 100) if cost_open else 0.0, 2)

    return {
        **pos,
        "current_price":   round(current_price, 4),
        "current_value":   round(current_price * cq, 2),
        "unrealized_pnl":  unreal,
        "total_pnl":       total_pnl,
        "total_pnl_pct":   total_pnl_pct,
    }
