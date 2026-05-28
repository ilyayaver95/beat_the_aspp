"""
auth_db.py
==========
SQLite database for authentication and per-user data.

Tables:
  users      - id, username (unique), password_hash (nullable for Google-only),
               email, google_sub (nullable, unique), created_at, last_login
  favorites  - user_id, ticker, created_at  (unique on (user_id, ticker))

All passwords are hashed with bcrypt. Every query uses ? parameters.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import bcrypt

DB_PATH = "data/users.db"

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    username            TEXT    NOT NULL UNIQUE,
    password_hash       TEXT,                   -- NULL for Google-only accounts
    email               TEXT    NOT NULL DEFAULT '',
    google_sub          TEXT    UNIQUE,         -- Google subject id, NULL for password accounts
    created_at          TEXT    NOT NULL,
    last_login          TEXT,
    telegram_bot_token  TEXT,
    telegram_chat_id    TEXT
);
CREATE TABLE IF NOT EXISTS favorites (
    user_id    INTEGER NOT NULL,
    ticker     TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    PRIMARY KEY (user_id, ticker),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id);
"""

# Columns added after the initial release. Applied as idempotent ALTER TABLE.
_USERS_MIGRATIONS = [
    ("telegram_bot_token", "TEXT"),
    ("telegram_chat_id",   "TEXT"),
]


@contextmanager
def _db(db_path: str = DB_PATH):
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
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


def init_db(db_path: str = DB_PATH) -> None:
    with _db(db_path) as conn:
        conn.executescript(_DDL)
        # Migrate older user DBs that pre-date later columns.
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        for col_name, col_type in _USERS_MIGRATIONS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")


# ── Password hashing ──────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── Validation ────────────────────────────────────────────────────

def _normalize_username(username: str) -> str:
    u = (username or "").strip().lower()
    if not u or len(u) < 3 or len(u) > 32:
        raise ValueError("Username must be 3-32 characters.")
    if not all(c.isalnum() or c in "._-" for c in u):
        raise ValueError("Username may only contain letters, digits, dot, underscore, hyphen.")
    return u


def _validate_password(password: str) -> None:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if len(password) > 128:
        raise ValueError("Password is too long.")


# ── User CRUD ─────────────────────────────────────────────────────

def register_user(
    username: str,
    password: str,
    email: str = "",
    db_path: str = DB_PATH,
) -> dict:
    """Create a new password-based user. Returns the user row as a dict."""
    username = _normalize_username(username)
    _validate_password(password)
    email = (email or "").strip().lower()[:200]
    pw_hash = _hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _db(db_path) as conn:
            cur = conn.execute(
                """INSERT INTO users (username, password_hash, email, created_at)
                   VALUES (?, ?, ?, ?)""",
                (username, pw_hash, email, now),
            )
            uid = cur.lastrowid
            row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
            return dict(row)
    except sqlite3.IntegrityError:
        raise ValueError("That username is already taken.")


def login_with_password(
    username: str,
    password: str,
    db_path: str = DB_PATH,
) -> dict | None:
    """Verify credentials and return the user row, or None on failure."""
    try:
        username = _normalize_username(username)
    except ValueError:
        return None
    with _db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        if not row:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), row["id"]),
        )
        return dict(row)


def upsert_google_user(
    google_sub: str,
    email: str,
    name: str = "",
    db_path: str = DB_PATH,
) -> dict:
    """Register or fetch a Google-authenticated user. Returns the user row."""
    google_sub = (google_sub or "").strip()
    if not google_sub:
        raise ValueError("Missing Google subject id.")
    email = (email or "").strip().lower()[:200]
    now = datetime.now(timezone.utc).isoformat()

    with _db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE google_sub=?", (google_sub,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE users SET last_login=?, email=? WHERE id=?",
                (now, email or row["email"], row["id"]),
            )
            return dict(row)

        # Derive a username from email/name. Make it unique.
        base = (email.split("@")[0] if email else (name or "user")).lower()
        base = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
        base = base.strip("._-") or "user"
        base = base[:24]
        candidate = base
        i = 1
        while conn.execute(
            "SELECT 1 FROM users WHERE username=?", (candidate,)
        ).fetchone() is not None:
            i += 1
            candidate = f"{base}{i}"[:32]

        cur = conn.execute(
            """INSERT INTO users (username, password_hash, email, google_sub, created_at, last_login)
               VALUES (?, NULL, ?, ?, ?, ?)""",
            (candidate, email, google_sub, now, now),
        )
        uid = cur.lastrowid
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return dict(row)


def get_user_by_id(user_id: int, db_path: str = DB_PATH) -> dict | None:
    with _db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (int(user_id),)).fetchone()
        return dict(row) if row else None


# ── Favorites ─────────────────────────────────────────────────────

def get_favorites(user_id: int, db_path: str = DB_PATH) -> list[str]:
    with _db(db_path) as conn:
        rows = conn.execute(
            "SELECT ticker FROM favorites WHERE user_id=? ORDER BY ticker",
            (int(user_id),),
        ).fetchall()
        return [r["ticker"] for r in rows]


def add_favorite(user_id: int, ticker: str, db_path: str = DB_PATH) -> None:
    ticker = (ticker or "").strip().upper()
    if not ticker or len(ticker) > 10:
        raise ValueError("Ticker must be 1-10 characters.")
    now = datetime.now(timezone.utc).isoformat()
    with _db(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO favorites (user_id, ticker, created_at)
               VALUES (?, ?, ?)""",
            (int(user_id), ticker, now),
        )


def remove_favorite(user_id: int, ticker: str, db_path: str = DB_PATH) -> None:
    ticker = (ticker or "").strip().upper()
    with _db(db_path) as conn:
        conn.execute(
            "DELETE FROM favorites WHERE user_id=? AND ticker=?",
            (int(user_id), ticker),
        )


def set_favorites(user_id: int, tickers: list[str], db_path: str = DB_PATH) -> None:
    """Replace the user's favorites list with the given ordered list."""
    now = datetime.now(timezone.utc).isoformat()
    cleaned = []
    for t in tickers:
        t = (t or "").strip().upper()
        if t and len(t) <= 10 and t not in cleaned:
            cleaned.append(t)
    with _db(db_path) as conn:
        conn.execute("DELETE FROM favorites WHERE user_id=?", (int(user_id),))
        conn.executemany(
            """INSERT OR IGNORE INTO favorites (user_id, ticker, created_at)
               VALUES (?, ?, ?)""",
            [(int(user_id), t, now) for t in cleaned],
        )


# ── Telegram credentials ──────────────────────────────────────────

def set_telegram_credentials(
    user_id: int,
    bot_token: str,
    chat_id: str,
    db_path: str = DB_PATH,
) -> None:
    """Save (and lightly validate) the user's Telegram bot token + chat id."""
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    if not bot_token or ":" not in bot_token or len(bot_token) > 200:
        raise ValueError("Bot token looks invalid. Expected format: 123456789:AA...")
    if not chat_id or len(chat_id) > 64:
        raise ValueError("Chat id is required (the number from getUpdates).")
    # chat_id may be negative for groups, but otherwise digits-only.
    cid_test = chat_id[1:] if chat_id.startswith("-") else chat_id
    if not cid_test.isdigit():
        raise ValueError("Chat id must be a number (e.g. 123456789 or -100123).")
    with _db(db_path) as conn:
        conn.execute(
            "UPDATE users SET telegram_bot_token=?, telegram_chat_id=? WHERE id=?",
            (bot_token, chat_id, int(user_id)),
        )


def clear_telegram_credentials(user_id: int, db_path: str = DB_PATH) -> None:
    with _db(db_path) as conn:
        conn.execute(
            "UPDATE users SET telegram_bot_token=NULL, telegram_chat_id=NULL WHERE id=?",
            (int(user_id),),
        )


def get_telegram_credentials(user_id: int, db_path: str = DB_PATH) -> tuple[str | None, str | None]:
    with _db(db_path) as conn:
        row = conn.execute(
            "SELECT telegram_bot_token, telegram_chat_id FROM users WHERE id=?",
            (int(user_id),),
        ).fetchone()
        if not row:
            return None, None
        return row["telegram_bot_token"], row["telegram_chat_id"]
