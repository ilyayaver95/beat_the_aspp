"""
migrate_to_db.py
================
One-shot migration from the old per-file SQLite layout into the new
unified database (Postgres on Cloud, SQLite locally — picked up via
DATABASE_URL or the data/local.db fallback in db.py).

Inputs scanned under ./data/:
  - users.db                        (users + favorites + tg credentials)
  - portfolio.db                    (legacy unscoped trades)
  - portfolio_local.db              (legacy unscoped trades, local-fallback path)
  - portfolio_u<N>.db               (per-user-id trade files)
  - portfolio_<sha20>.db            (per-email-hash trade files)
  - portfolio_baseline*.db          (same naming for baseline data)
  - favorites.json                  (legacy global favorites — attached to user 1)

Idempotent: re-running won't duplicate rows. Pass --dry-run to see what
would happen without writing.

Run with:
  python migrate_to_db.py             # writes
  python migrate_to_db.py --dry-run   # report only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

from sqlalchemy import text

import db as _db
import auth_db
import portfolio_db
import portfolio_baseline_db

DATA_DIR = "data"


# ── Helpers ───────────────────────────────────────────────────────

def _open_sqlite(path: str) -> sqlite3.Connection | None:
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def _row_exists(conn, sql: str, params: dict) -> bool:
    return conn.execute(text(sql), params).first() is not None


def _parse_iso(s) -> datetime:
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    if not s:
        return datetime.now(timezone.utc)
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


# ── 1. Users + favorites + telegram from old data/users.db ────────

def migrate_users(dry_run: bool) -> dict[int, int]:
    """
    Copy users from old data/users.db into the active engine.
    Returns a mapping of {old_user_id: new_user_id} so subsequent steps
    can re-key portfolio rows.
    """
    src_path = os.path.join(DATA_DIR, "users.db")
    src = _open_sqlite(src_path)
    if src is None:
        print(f"[users] {src_path} not found — skipping")
        return {}

    mapping: dict[int, int] = {}
    eng = _db.get_engine()
    with src, eng.begin() as conn:
        old_users = src.execute("SELECT * FROM users").fetchall()
        print(f"[users] found {len(old_users)} user(s) in {src_path}")

        for u in old_users:
            old_id = int(u["id"])
            username = u["username"]
            existing = conn.execute(
                text("SELECT id FROM users WHERE username=:u"),
                {"u": username},
            ).first()
            if existing:
                new_id = int(existing[0])
                mapping[old_id] = new_id
                print(f"[users] '{username}' already present -> id {new_id}")
                continue

            params = {
                "username":      username,
                "pw":            u["password_hash"],
                "email":         u["email"] or "",
                "sub":           u["google_sub"],
                "created":       _parse_iso(u["created_at"]),
                "last":          _parse_iso(u["last_login"]) if u["last_login"] else None,
                "tg_tok":        u["telegram_bot_token"] if "telegram_bot_token" in u.keys() else None,
                "tg_cid":        u["telegram_chat_id"]   if "telegram_chat_id"   in u.keys() else None,
            }
            if dry_run:
                # In dry-run, reuse the OLD id so the cascading row counts
                # (favorites etc.) report something useful.
                new_id = old_id
                print(f"[users] would insert '{username}' (preview id={old_id})")
            else:
                if eng.dialect.name == "postgresql":
                    res = conn.execute(
                        text(
                            "INSERT INTO users (username, password_hash, email, "
                            "google_sub, created_at, last_login, telegram_bot_token, "
                            "telegram_chat_id) VALUES "
                            "(:username, :pw, :email, :sub, :created, :last, :tg_tok, :tg_cid) "
                            "RETURNING id"
                        ),
                        params,
                    )
                    new_id = int(res.scalar_one())
                else:
                    res = conn.execute(
                        text(
                            "INSERT INTO users (username, password_hash, email, "
                            "google_sub, created_at, last_login, telegram_bot_token, "
                            "telegram_chat_id) VALUES "
                            "(:username, :pw, :email, :sub, :created, :last, :tg_tok, :tg_cid)"
                        ),
                        params,
                    )
                    new_id = int(res.lastrowid)
                print(f"[users] inserted '{username}' -> id {new_id}")
            mapping[old_id] = new_id

        # Favorites (old DB) → new favorites table, keyed by NEW user_id.
        if _table_exists(src, "favorites"):
            old_favs = src.execute("SELECT * FROM favorites").fetchall()
            inserted = 0
            for f in old_favs:
                new_uid = mapping.get(int(f["user_id"]))
                if new_uid is None:
                    continue
                if _row_exists(
                    conn,
                    "SELECT 1 FROM favorites WHERE user_id=:u AND ticker=:t",
                    {"u": new_uid, "t": f["ticker"]},
                ):
                    continue
                if not dry_run:
                    conn.execute(
                        text(
                            "INSERT INTO favorites (user_id, ticker, created_at) "
                            "VALUES (:u, :t, :ts)"
                        ),
                        {"u": new_uid, "t": f["ticker"], "ts": _parse_iso(f["created_at"])},
                    )
                inserted += 1
            print(f"[favorites] {'would insert' if dry_run else 'inserted'} {inserted} rows")

    return mapping


# ── 2. Favorites from the legacy global JSON ──────────────────────

def migrate_legacy_favorites_json(mapping: dict[int, int], dry_run: bool) -> None:
    path = os.path.join(DATA_DIR, "favorites.json")
    if not os.path.exists(path):
        return
    if not mapping:
        print(f"[favorites.json] no user mapping — skipping (no user to attach to)")
        return
    new_uid = next(iter(mapping.values()))
    try:
        with open(path, encoding="utf-8") as f:
            tickers = [str(t).strip().upper() for t in json.load(f) if str(t).strip()]
    except Exception as e:
        print(f"[favorites.json] could not read: {e}")
        return
    print(f"[favorites.json] attaching {len(tickers)} legacy favorite(s) to user_id={new_uid}")
    if not dry_run:
        for t in tickers:
            try:
                auth_db.add_favorite(new_uid, t)
            except Exception as e:
                print(f"  could not add {t!r}: {e}")


# ── 3. Portfolio (trades) files ──────────────────────────────────

_PORTFOLIO_GLOBS = [
    ("portfolio_u*.db",       "uN"),
    ("portfolio_local.db",    "local"),
    ("portfolio.db",          "legacy"),
    ("portfolio_*.db",        "hash"),   # catch-all for sha-hashed names
]


def _resolve_user_for_file(filename: str, mapping: dict[int, int]) -> int | None:
    """
    Map a per-user portfolio file to a NEW user_id.
      portfolio_uN.db          -> mapping[N]
      portfolio_local.db       -> first user (most common: ilya)
      portfolio.db (no suffix) -> first user
      portfolio_<sha>.db       -> first user  (we can't reverse SHA cheaply,
                                  but locally there's only ever 1 user, so
                                  this is fine — confirm before running on
                                  a multi-user laptop.)
    """
    base = os.path.basename(filename).removesuffix(".db")
    if base.startswith("portfolio_baseline_"):
        suffix = base[len("portfolio_baseline_"):]
    elif base.startswith("portfolio_"):
        suffix = base[len("portfolio_"):]
    else:
        suffix = ""

    if suffix.startswith("u") and suffix[1:].isdigit():
        return mapping.get(int(suffix[1:]))
    if not mapping:
        return None
    # Fallback: only one local user, attach everything to them.
    return next(iter(mapping.values()))


def migrate_trades_files(mapping: dict[int, int], dry_run: bool) -> None:
    seen: set[str] = set()
    candidates: list[str] = []
    for pattern in [
        "portfolio_u*.db", "portfolio_local.db", "portfolio.db",
    ]:
        candidates.extend(glob.glob(os.path.join(DATA_DIR, pattern)))
    # also catch portfolio_<sha>.db (but exclude baseline + the new file)
    for path in glob.glob(os.path.join(DATA_DIR, "portfolio_*.db")):
        name = os.path.basename(path)
        if name.startswith("portfolio_baseline"):
            continue
        candidates.append(path)

    eng = _db.get_engine()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        new_uid = _resolve_user_for_file(path, mapping)
        if new_uid is None:
            print(f"[trades] {path}: cannot resolve owner — skipping")
            continue

        src = _open_sqlite(path)
        if src is None or not _table_exists(src, "trades"):
            if src: src.close()
            continue
        rows = src.execute("SELECT * FROM trades").fetchall()
        src.close()
        if not rows:
            print(f"[trades] {path}: empty")
            continue

        inserted = 0
        with eng.begin() as conn:
            for r in rows:
                if _row_exists(
                    conn,
                    "SELECT 1 FROM trades WHERE user_id=:u AND ticker=:t "
                    "AND action=:a AND trade_date=:d AND quantity=:q "
                    "AND price_per_share=:p",
                    {"u": new_uid, "t": r["ticker"], "a": r["action"],
                     "d": r["trade_date"], "q": r["quantity"],
                     "p": r["price_per_share"]},
                ):
                    continue
                if not dry_run:
                    conn.execute(
                        text(
                            "INSERT INTO trades "
                            "(user_id, ticker, action, trade_date, quantity, "
                            " price_per_share, notes, created_at) VALUES "
                            "(:uid, :t, :a, :d, :q, :p, :n, :ts)"
                        ),
                        {
                            "uid": new_uid, "t": r["ticker"], "a": r["action"],
                            "d": r["trade_date"], "q": r["quantity"],
                            "p": r["price_per_share"], "n": r["notes"] or "",
                            "ts": _parse_iso(r["created_at"]),
                        },
                    )
                inserted += 1
        print(f"[trades] {path} -> user_id={new_uid}: "
              f"{'would insert' if dry_run else 'inserted'} {inserted}/{len(rows)} new rows")


# ── 4. Baseline files (meta, positions, trades, transfers) ────────

def migrate_baseline_files(mapping: dict[int, int], dry_run: bool) -> None:
    candidates = []
    for pattern in [
        "portfolio_baseline.db",
        "portfolio_baseline_local.db",
        "portfolio_baseline_u*.db",
        "portfolio_baseline_*.db",
    ]:
        candidates.extend(glob.glob(os.path.join(DATA_DIR, pattern)))
    candidates = list(dict.fromkeys(candidates))  # dedupe, keep order

    eng = _db.get_engine()
    for path in candidates:
        new_uid = _resolve_user_for_file(path, mapping)
        if new_uid is None:
            print(f"[baseline] {path}: cannot resolve owner — skipping")
            continue
        src = _open_sqlite(path)
        if src is None:
            continue

        with eng.begin() as conn:
            # meta
            if _table_exists(src, "baseline_meta"):
                meta_rows = src.execute(
                    "SELECT key, value FROM baseline_meta"
                ).fetchall()
                ins = 0
                for m in meta_rows:
                    if _row_exists(
                        conn,
                        "SELECT 1 FROM baseline_meta WHERE user_id=:u AND key=:k",
                        {"u": new_uid, "k": m["key"]},
                    ):
                        continue
                    if not dry_run:
                        conn.execute(
                            text(
                                "INSERT INTO baseline_meta (user_id, key, value) "
                                "VALUES (:u, :k, :v)"
                            ),
                            {"u": new_uid, "k": m["key"], "v": m["value"]},
                        )
                    ins += 1
                if meta_rows:
                    print(f"[baseline_meta] {path} -> user_id={new_uid}: "
                          f"{'would insert' if dry_run else 'inserted'} "
                          f"{ins}/{len(meta_rows)} new rows")

            # positions
            if _table_exists(src, "baseline_positions"):
                pos_rows = src.execute(
                    "SELECT ticker, quantity, avg_cost_per_share, created_at "
                    "FROM baseline_positions"
                ).fetchall()
                ins = 0
                for p in pos_rows:
                    if _row_exists(
                        conn,
                        "SELECT 1 FROM baseline_positions WHERE user_id=:u AND ticker=:t",
                        {"u": new_uid, "t": p["ticker"]},
                    ):
                        continue
                    if not dry_run:
                        conn.execute(
                            text(
                                "INSERT INTO baseline_positions "
                                "(user_id, ticker, quantity, avg_cost_per_share, created_at) "
                                "VALUES (:u, :t, :q, :a, :ts)"
                            ),
                            {"u": new_uid, "t": p["ticker"], "q": p["quantity"],
                             "a": p["avg_cost_per_share"], "ts": _parse_iso(p["created_at"])},
                        )
                    ins += 1
                if pos_rows:
                    print(f"[baseline_positions] {path} -> user_id={new_uid}: "
                          f"{'would insert' if dry_run else 'inserted'} "
                          f"{ins}/{len(pos_rows)} new rows")

            # trades
            if _table_exists(src, "baseline_trades"):
                bt_rows = src.execute("SELECT * FROM baseline_trades").fetchall()
                ins = 0
                for r in bt_rows:
                    if _row_exists(
                        conn,
                        "SELECT 1 FROM baseline_trades WHERE user_id=:u AND ticker=:t "
                        "AND action=:a AND trade_date=:d AND quantity=:q "
                        "AND price_per_share=:p",
                        {"u": new_uid, "t": r["ticker"], "a": r["action"],
                         "d": r["trade_date"], "q": r["quantity"], "p": r["price_per_share"]},
                    ):
                        continue
                    if not dry_run:
                        conn.execute(
                            text(
                                "INSERT INTO baseline_trades "
                                "(user_id, ticker, action, trade_date, quantity, "
                                " price_per_share, notes, created_at) VALUES "
                                "(:u, :t, :a, :d, :q, :p, :n, :ts)"
                            ),
                            {"u": new_uid, "t": r["ticker"], "a": r["action"],
                             "d": r["trade_date"], "q": r["quantity"],
                             "p": r["price_per_share"], "n": r["notes"] or "",
                             "ts": _parse_iso(r["created_at"])},
                        )
                    ins += 1
                if bt_rows:
                    print(f"[baseline_trades] {path} -> user_id={new_uid}: "
                          f"{'would insert' if dry_run else 'inserted'} "
                          f"{ins}/{len(bt_rows)} new rows")

            # transfers
            if _table_exists(src, "baseline_transfers"):
                xrows = src.execute("SELECT * FROM baseline_transfers").fetchall()
                ins = 0
                for r in xrows:
                    if _row_exists(
                        conn,
                        "SELECT 1 FROM baseline_transfers WHERE user_id=:u AND direction=:d "
                        "AND transfer_date=:dt AND amount=:a",
                        {"u": new_uid, "d": r["direction"],
                         "dt": r["transfer_date"], "a": r["amount"]},
                    ):
                        continue
                    if not dry_run:
                        conn.execute(
                            text(
                                "INSERT INTO baseline_transfers "
                                "(user_id, direction, transfer_date, amount, notes, created_at) "
                                "VALUES (:u, :d, :dt, :a, :n, :ts)"
                            ),
                            {"u": new_uid, "d": r["direction"], "dt": r["transfer_date"],
                             "a": r["amount"], "n": r["notes"] or "",
                             "ts": _parse_iso(r["created_at"])},
                        )
                    ins += 1
                if xrows:
                    print(f"[baseline_transfers] {path} -> user_id={new_uid}: "
                          f"{'would insert' if dry_run else 'inserted'} "
                          f"{ins}/{len(xrows)} new rows")

        src.close()


# ── Entry point ───────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without writing.",
    )
    args = parser.parse_args()

    target = "Postgres" if _db.is_postgres() else "local SQLite (data/local.db)"
    print(f"Target: {target}")
    if args.dry_run:
        print("DRY RUN — no rows will be inserted.\n")

    _db.init_all()
    mapping = migrate_users(args.dry_run)
    migrate_legacy_favorites_json(mapping, args.dry_run)
    migrate_trades_files(mapping, args.dry_run)
    migrate_baseline_files(mapping, args.dry_run)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
