"""
portfolio_baseline_db.py
========================
Baseline + post-baseline trades + cash transfers for the Baseline page.

Multi-tenant via a user_id column on every row. Runs on Postgres on
Streamlit Cloud and SQLite locally (single shared engine from db.py).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Table, Column, Integer, String, Text, Float, DateTime,
    UniqueConstraint, Index, text,
)

from db import get_engine, metadata


# ── Schema ────────────────────────────────────────────────────────

baseline_meta = Table(
    "baseline_meta", metadata,
    Column("user_id", Integer, nullable=False),
    Column("key",   String(32), nullable=False),
    Column("value", Text, nullable=False),
    UniqueConstraint("user_id", "key", name="uq_baseline_meta_user_key"),
    Index("idx_baseline_meta_user", "user_id"),
)

baseline_positions = Table(
    "baseline_positions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False),
    Column("ticker", String(10), nullable=False),
    Column("quantity", Float, nullable=False),
    Column("avg_cost_per_share", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("user_id", "ticker", name="uq_baseline_positions_user_ticker"),
    Index("idx_baseline_positions_user", "user_id"),
)

baseline_trades = Table(
    "baseline_trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False),
    Column("ticker", String(10), nullable=False),
    Column("action", String(4), nullable=False),
    Column("trade_date", String(10), nullable=False),
    Column("quantity", Float, nullable=False),
    Column("price_per_share", Float, nullable=False),
    Column("notes", Text, nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("idx_baseline_trades_user", "user_id"),
    Index("idx_baseline_trades_user_ticker", "user_id", "ticker"),
)

baseline_transfers = Table(
    "baseline_transfers", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False),
    Column("direction", String(3), nullable=False),    # IN / OUT
    Column("transfer_date", String(10), nullable=False),
    Column("amount", Float, nullable=False),
    Column("notes", Text, nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("idx_baseline_transfers_user", "user_id"),
)


def init_db() -> None:
    metadata.create_all(
        get_engine(),
        tables=[baseline_meta, baseline_positions,
                baseline_trades, baseline_transfers],
    )


# ── Meta (free_cash, as_of_date) ──────────────────────────────────

def get_meta(user_id: int) -> dict:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text("SELECT key, value FROM baseline_meta WHERE user_id=:uid"),
            {"uid": int(user_id)},
        ).all()
    meta = {r[0]: r[1] for r in rows}
    return {
        "free_cash":  float(meta.get("free_cash", "0") or 0),
        "as_of_date": meta.get("as_of_date", ""),
    }


def set_meta(user_id: int, free_cash: float, as_of_date: str) -> None:
    if free_cash < 0:
        raise ValueError("Free cash cannot be negative.")
    try:
        datetime.strptime(as_of_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("As-of date must be in YYYY-MM-DD format.")
    rows = [
        {"uid": int(user_id), "k": "free_cash",  "v": f"{float(free_cash):.4f}"},
        {"uid": int(user_id), "k": "as_of_date", "v": as_of_date},
    ]
    sql = text(
        "INSERT INTO baseline_meta (user_id, key, value) "
        "VALUES (:uid, :k, :v) "
        "ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value"
    )
    with get_engine().begin() as conn:
        for p in rows:
            conn.execute(sql, p)


# ── Baseline positions ────────────────────────────────────────────

def _validate_position(ticker: str, quantity: float, avg_cost: float) -> str:
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 10:
        raise ValueError("Ticker must be 1-10 characters.")
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        raise ValueError("Quantity must be a positive number.")
    if not isinstance(avg_cost, (int, float)) or avg_cost <= 0:
        raise ValueError("Avg cost per share must be a positive number.")
    return ticker


def upsert_position(user_id: int, ticker: str, quantity: float, avg_cost: float) -> None:
    ticker = _validate_position(ticker, quantity, avg_cost)
    sql = text(
        "INSERT INTO baseline_positions "
        "  (user_id, ticker, quantity, avg_cost_per_share, created_at) "
        "VALUES (:uid, :ticker, :qty, :avg, :now) "
        "ON CONFLICT (user_id, ticker) DO UPDATE SET "
        "  quantity = EXCLUDED.quantity, "
        "  avg_cost_per_share = EXCLUDED.avg_cost_per_share"
    )
    with get_engine().begin() as conn:
        conn.execute(sql, {
            "uid": int(user_id), "ticker": ticker,
            "qty": float(quantity), "avg": float(avg_cost),
            "now": datetime.now(timezone.utc),
        })


def delete_position(user_id: int, ticker: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "DELETE FROM baseline_positions "
                "WHERE user_id=:uid AND ticker=:t"
            ),
            {"uid": int(user_id), "t": ticker.strip().upper()},
        )


def get_positions(user_id: int) -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT ticker, quantity, avg_cost_per_share "
                "FROM baseline_positions WHERE user_id=:uid ORDER BY ticker ASC"
            ),
            {"uid": int(user_id)},
        ).all()
    return [dict(r._mapping) for r in rows]


def replace_all_positions(user_id: int, positions: list[dict]) -> None:
    cleaned: list[dict] = []
    seen: set[str] = set()
    for p in positions:
        tkr = _validate_position(p["ticker"], p["quantity"], p["avg_cost_per_share"])
        if tkr in seen:
            raise ValueError(f"Duplicate ticker '{tkr}' in baseline.")
        seen.add(tkr)
        cleaned.append({
            "uid": int(user_id), "ticker": tkr,
            "qty": float(p["quantity"]),
            "avg": float(p["avg_cost_per_share"]),
            "now": datetime.now(timezone.utc),
        })
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM baseline_positions WHERE user_id=:uid"),
            {"uid": int(user_id)},
        )
        if cleaned:
            conn.execute(
                text(
                    "INSERT INTO baseline_positions "
                    "  (user_id, ticker, quantity, avg_cost_per_share, created_at) "
                    "VALUES (:uid, :ticker, :qty, :avg, :now)"
                ),
                cleaned,
            )


# ── Value / Total-Return conversions (no DB) ──────────────────────

def value_return_to_qty_cost(value: float, total_return_pct: float,
                             ref_price: float) -> tuple[float, float]:
    if ref_price <= 0:
        raise ValueError("Live price unavailable — cannot convert value/return.")
    if value <= 0:
        raise ValueError("Value must be a positive number.")
    if total_return_pct <= -100:
        raise ValueError("Total return must be greater than -100%.")
    qty      = value / ref_price
    avg_cost = ref_price / (1.0 + total_return_pct / 100.0)
    return qty, avg_cost


def qty_cost_to_value_return(qty: float, avg_cost: float,
                             ref_price: float) -> tuple[float, float]:
    value  = qty * ref_price
    ret_pc = ((ref_price - avg_cost) / avg_cost * 100.0) if avg_cost > 0 else 0.0
    return value, ret_pc


def replace_all_positions_from_value_return(
    user_id: int, rows: list[dict], prices: dict[str, float],
) -> None:
    cleaned: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        tkr = str(r["ticker"]).strip().upper()
        if not tkr:
            continue
        if tkr in seen:
            raise ValueError(f"Duplicate ticker '{tkr}' in baseline.")
        seen.add(tkr)
        price = float(prices.get(tkr, 0.0) or 0.0)
        qty, avg_cost = value_return_to_qty_cost(
            float(r["value"]), float(r["total_return_pct"]), price
        )
        _validate_position(tkr, qty, avg_cost)
        cleaned.append({
            "uid": int(user_id), "ticker": tkr,
            "qty": qty, "avg": avg_cost,
            "now": datetime.now(timezone.utc),
        })
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM baseline_positions WHERE user_id=:uid"),
            {"uid": int(user_id)},
        )
        if cleaned:
            conn.execute(
                text(
                    "INSERT INTO baseline_positions "
                    "  (user_id, ticker, quantity, avg_cost_per_share, created_at) "
                    "VALUES (:uid, :ticker, :qty, :avg, :now)"
                ),
                cleaned,
            )


# ── Post-baseline trades ──────────────────────────────────────────

def _validate_trade(ticker, action, trade_date, quantity, price_per_share):
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


def add_trade(user_id: int, ticker, action, trade_date,
              quantity, price_per_share, notes: str = "") -> int:
    ticker, action = _validate_trade(ticker, action, trade_date, quantity, price_per_share)
    notes = (notes or "").strip()[:500]
    eng = get_engine()
    sql_pg = text(
        "INSERT INTO baseline_trades "
        "  (user_id, ticker, action, trade_date, quantity, price_per_share, notes, created_at) "
        "VALUES (:uid, :ticker, :action, :date, :qty, :price, :notes, :now) "
        "RETURNING id"
    )
    sql_lite = text(
        "INSERT INTO baseline_trades "
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


def update_trade(trade_id, user_id: int, ticker, action, trade_date,
                 quantity, price_per_share, notes: str = "") -> None:
    ticker, action = _validate_trade(ticker, action, trade_date, quantity, price_per_share)
    notes = (notes or "").strip()[:500]
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE baseline_trades SET "
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
            text("DELETE FROM baseline_trades WHERE id=:id AND user_id=:uid"),
            {"id": int(trade_id), "uid": int(user_id)},
        )


def get_all_trades(user_id: int) -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM baseline_trades WHERE user_id=:uid "
                "ORDER BY trade_date DESC, id DESC"
            ),
            {"uid": int(user_id)},
        ).all()
    return [dict(r._mapping) for r in rows]


def get_trades_for_ticker(user_id: int, ticker: str) -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM baseline_trades "
                "WHERE user_id=:uid AND ticker=:t "
                "ORDER BY trade_date ASC, id ASC"
            ),
            {"uid": int(user_id), "t": ticker.strip().upper()},
        ).all()
    return [dict(r._mapping) for r in rows]


# ── Cash transfers ────────────────────────────────────────────────

def _validate_transfer(direction: str, transfer_date: str, amount: float) -> str:
    direction = direction.strip().upper()
    if direction not in ("IN", "OUT"):
        raise ValueError("Direction must be IN or OUT.")
    try:
        datetime.strptime(transfer_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Transfer date must be in YYYY-MM-DD format.")
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise ValueError("Amount must be a positive number.")
    return direction


def add_transfer(user_id: int, direction: str, transfer_date: str,
                 amount: float, notes: str = "") -> int:
    direction = _validate_transfer(direction, transfer_date, amount)
    notes = (notes or "").strip()[:500]
    eng = get_engine()
    sql_pg = text(
        "INSERT INTO baseline_transfers "
        "  (user_id, direction, transfer_date, amount, notes, created_at) "
        "VALUES (:uid, :dir, :date, :amt, :notes, :now) "
        "RETURNING id"
    )
    sql_lite = text(
        "INSERT INTO baseline_transfers "
        "  (user_id, direction, transfer_date, amount, notes, created_at) "
        "VALUES (:uid, :dir, :date, :amt, :notes, :now)"
    )
    params = {
        "uid": int(user_id), "dir": direction, "date": transfer_date,
        "amt": float(amount), "notes": notes,
        "now": datetime.now(timezone.utc),
    }
    with eng.begin() as conn:
        if eng.dialect.name == "postgresql":
            return int(conn.execute(sql_pg, params).scalar_one())
        return int(conn.execute(sql_lite, params).lastrowid)


def update_transfer(transfer_id: int, user_id: int, direction: str,
                    transfer_date: str, amount: float,
                    notes: str = "") -> None:
    direction = _validate_transfer(direction, transfer_date, amount)
    notes = (notes or "").strip()[:500]
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE baseline_transfers SET "
                "  direction=:dir, transfer_date=:date, amount=:amt, notes=:notes "
                "WHERE id=:id AND user_id=:uid"
            ),
            {
                "dir": direction, "date": transfer_date,
                "amt": float(amount), "notes": notes,
                "id": int(transfer_id), "uid": int(user_id),
            },
        )


def delete_transfer(transfer_id: int, user_id: int) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM baseline_transfers WHERE id=:id AND user_id=:uid"),
            {"id": int(transfer_id), "uid": int(user_id)},
        )


def get_all_transfers(user_id: int) -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM baseline_transfers WHERE user_id=:uid "
                "ORDER BY transfer_date DESC, id DESC"
            ),
            {"uid": int(user_id)},
        ).all()
    return [dict(r._mapping) for r in rows]


def get_transfer_totals(user_id: int) -> dict:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text(
                "SELECT direction, COALESCE(SUM(amount), 0) AS total "
                "FROM baseline_transfers WHERE user_id=:uid "
                "GROUP BY direction"
            ),
            {"uid": int(user_id)},
        ).all()
    totals = {r[0]: float(r[1]) for r in rows}
    total_in  = totals.get("IN", 0.0)
    total_out = totals.get("OUT", 0.0)
    return {
        "total_in":  round(total_in, 2),
        "total_out": round(total_out, 2),
        "net":       round(total_in - total_out, 2),
    }


# ── Combined P&L (no DB) ──────────────────────────────────────────

def compute_combined_position(baseline_qty: float, baseline_avg_cost: float,
                              trades: list[dict]) -> dict:
    qty       = float(baseline_qty)
    avg_cost  = float(baseline_avg_cost)
    realized  = 0.0
    cash_flow = 0.0

    total_buy_qty  = qty
    total_sell_qty = 0.0
    total_buy_cost = qty * avg_cost

    for t in trades:
        q = float(t["quantity"])
        p = float(t["price_per_share"])
        if t["action"] == "BUY":
            total_buy_qty  += q
            total_buy_cost += q * p
            new_qty   = qty + q
            avg_cost  = ((qty * avg_cost) + (q * p)) / new_qty if new_qty > 0 else 0.0
            qty       = new_qty
            cash_flow -= q * p
        else:
            sell_qty_eff = qty if q > qty + 1e-9 else q
            realized       += (p - avg_cost) * sell_qty_eff
            qty            -= q
            total_sell_qty += q
            cash_flow      += q * p

    ticker = trades[0]["ticker"] if trades else ""
    return {
        "ticker":                ticker,
        "total_buy_qty":         round(total_buy_qty, 6),
        "total_sell_qty":        round(total_sell_qty, 6),
        "current_qty":           round(qty, 6),
        "avg_cost_basis":        round(avg_cost, 4),
        "total_buy_cost":        round(total_buy_cost, 2),
        "total_sell_proceeds":   round(sum(
            t["quantity"] * t["price_per_share"]
            for t in trades if t["action"] == "SELL"
        ), 2),
        "cost_of_open_position": round(max(qty, 0) * avg_cost, 2),
        "realized_pnl":          round(realized, 2),
        "cash_flow":             round(cash_flow, 2),
        "current_price":         0.0,
        "current_value":         0.0,
        "unrealized_pnl":        0.0,
        "total_pnl":             round(realized, 2),
        "total_pnl_pct":         0.0,
    }


def enrich_with_price(pos: dict, current_price: float) -> dict:
    cq        = max(pos["current_qty"], 0.0)
    avg       = pos["avg_cost_basis"]
    unreal    = round((current_price - avg) * cq, 2) if cq > 0 else 0.0
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
