"""
auth_db.py
==========
Users + per-user favorites and Telegram credentials.

Runs on Postgres in production (Streamlit Cloud + DATABASE_URL) and on
SQLite locally. All SQL is portable between the two dialects.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from sqlalchemy import (
    Table, Column, Integer, String, Text, DateTime, ForeignKey,
    UniqueConstraint, Index, text,
)

from db import get_engine, metadata


# ── Schema (registered on the shared MetaData) ────────────────────

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(64), nullable=False, unique=True),
    Column("password_hash", Text),                       # NULL for Google-only accounts
    Column("email", String(200), nullable=False, server_default=""),
    Column("google_sub", String(128), unique=True),      # NULL for password accounts
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_login", DateTime(timezone=True)),
    Column("telegram_bot_token", Text),
    Column("telegram_chat_id", String(64)),
)

favorites = Table(
    "favorites", metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("ticker", String(10), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("user_id", "ticker", name="uq_favorites_user_ticker"),
    Index("idx_favorites_user", "user_id"),
)

password_reset_codes = Table(
    "password_reset_codes", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("code_hash", String(64), nullable=False),     # sha256 hex digest
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("used_at", DateTime(timezone=True)),          # NULL until consumed
    Index("idx_pwreset_user", "user_id"),
)

RESET_CODE_TTL = timedelta(minutes=15)
RESET_REQUEST_COOLDOWN = timedelta(seconds=60)


def init_db() -> None:
    """Create users/favorites tables if missing."""
    metadata.create_all(get_engine(), tables=[users, favorites])


# ── Password hashing ──────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str | None) -> bool:
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

def _row_to_dict(row) -> dict | None:
    if row is None:
        return None
    return dict(row._mapping)


def register_user(username: str, password: str, email: str = "") -> dict:
    """Create a new password-based user. Returns the user row as a dict."""
    username = _normalize_username(username)
    _validate_password(password)
    email = (email or "").strip().lower()[:200]
    pw_hash = _hash_password(password)
    now = datetime.now(timezone.utc)

    eng = get_engine()
    with eng.begin() as conn:
        existing = conn.execute(
            text("SELECT 1 FROM users WHERE username = :u"), {"u": username}
        ).first()
        if existing:
            raise ValueError("That username is already taken.")

        res = conn.execute(
            text(
                "INSERT INTO users (username, password_hash, email, created_at) "
                "VALUES (:username, :pw, :email, :now) "
                "RETURNING id"
            ) if eng.dialect.name == "postgresql" else
            text(
                "INSERT INTO users (username, password_hash, email, created_at) "
                "VALUES (:username, :pw, :email, :now)"
            ),
            {"username": username, "pw": pw_hash, "email": email, "now": now},
        )
        if eng.dialect.name == "postgresql":
            uid = res.scalar_one()
        else:
            uid = res.lastrowid

        row = conn.execute(
            text("SELECT * FROM users WHERE id = :id"), {"id": uid}
        ).first()
        return _row_to_dict(row)


def login_with_password(username: str, password: str) -> dict | None:
    """Verify credentials and return the user row, or None on failure."""
    try:
        username = _normalize_username(username)
    except ValueError:
        return None
    with get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE username = :u"), {"u": username}
        ).first()
        if row is None:
            return None
        d = _row_to_dict(row)
        if not _verify_password(password, d.get("password_hash")):
            return None
        conn.execute(
            text("UPDATE users SET last_login = :now WHERE id = :id"),
            {"now": datetime.now(timezone.utc), "id": d["id"]},
        )
        d["last_login"] = datetime.now(timezone.utc)
        return d


def upsert_google_user(google_sub: str, email: str = "", name: str = "") -> dict:
    """Register or fetch a Google-authenticated user."""
    google_sub = (google_sub or "").strip()
    if not google_sub:
        raise ValueError("Missing Google subject id.")
    email = (email or "").strip().lower()[:200]
    now = datetime.now(timezone.utc)

    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE google_sub = :sub"),
            {"sub": google_sub},
        ).first()
        if row:
            d = _row_to_dict(row)
            new_email = email or d["email"]
            conn.execute(
                text(
                    "UPDATE users SET last_login = :now, email = :email "
                    "WHERE id = :id"
                ),
                {"now": now, "email": new_email, "id": d["id"]},
            )
            d["last_login"] = now
            d["email"] = new_email
            return d

        # Derive a unique username from the email local part / name.
        base = (email.split("@")[0] if email else (name or "user")).lower()
        base = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
        base = (base.strip("._-") or "user")[:24]
        candidate = base
        i = 1
        while conn.execute(
            text("SELECT 1 FROM users WHERE username = :u"), {"u": candidate}
        ).first():
            i += 1
            candidate = f"{base}{i}"[:32]

        res = conn.execute(
            text(
                "INSERT INTO users "
                "  (username, password_hash, email, google_sub, created_at, last_login) "
                "VALUES (:username, NULL, :email, :sub, :now, :now) "
                "RETURNING id"
            ) if eng.dialect.name == "postgresql" else
            text(
                "INSERT INTO users "
                "  (username, password_hash, email, google_sub, created_at, last_login) "
                "VALUES (:username, NULL, :email, :sub, :now, :now)"
            ),
            {"username": candidate, "email": email, "sub": google_sub, "now": now},
        )
        if eng.dialect.name == "postgresql":
            uid = res.scalar_one()
        else:
            uid = res.lastrowid

        new = conn.execute(
            text("SELECT * FROM users WHERE id = :id"), {"id": uid}
        ).first()
        return _row_to_dict(new)


def get_user_by_id(user_id: int) -> dict | None:
    with get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE id = :id"), {"id": int(user_id)}
        ).first()
        return _row_to_dict(row)


def get_user_by_email(email: str) -> dict | None:
    email = (email or "").strip().lower()
    if not email:
        return None
    with get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE email = :e"), {"e": email}
        ).first()
        return _row_to_dict(row)


# ── Password reset ────────────────────────────────────────────────

def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _generate_code() -> str:
    """Six-digit numeric code, zero-padded. ~1M combinations is fine
    given 15-min TTL + single-use + rate limit."""
    return f"{secrets.randbelow(1_000_000):06d}"


def create_password_reset_code(email: str) -> str | None:
    """Issue a fresh reset code for the user with this email.

    Returns the plaintext code so the caller can email it, or None if
    the email doesn't match any user OR the user requested one in the
    last RESET_REQUEST_COOLDOWN seconds (silent — anti-enumeration is
    the UI layer's job).
    """
    user = get_user_by_email(email)
    if user is None:
        return None

    now = datetime.now(timezone.utc)
    with get_engine().begin() as conn:
        recent = conn.execute(
            text(
                "SELECT created_at FROM password_reset_codes "
                "WHERE user_id = :uid AND used_at IS NULL "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"uid": user["id"]},
        ).first()
        if recent is not None:
            created_at = recent[0]
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if now - created_at < RESET_REQUEST_COOLDOWN:
                return None

        code = _generate_code()
        conn.execute(
            text(
                "INSERT INTO password_reset_codes "
                "  (user_id, code_hash, created_at, expires_at) "
                "VALUES (:uid, :h, :now, :exp)"
            ),
            {
                "uid": user["id"],
                "h": _hash_code(code),
                "now": now,
                "exp": now + RESET_CODE_TTL,
            },
        )
        return code


def reset_password_with_code(email: str, code: str, new_password: str) -> bool:
    """Consume a valid reset code and update the user's password.

    Returns True on success, False if the email/code combo is unknown,
    expired, or already used. Raises ValueError if new_password fails
    validation (so the UI can show the specific reason).
    """
    _validate_password(new_password)
    user = get_user_by_email(email)
    if user is None:
        return False

    code = (code or "").strip()
    if not code:
        return False
    code_hash = _hash_code(code)
    now = datetime.now(timezone.utc)

    with get_engine().begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, expires_at, used_at FROM password_reset_codes "
                "WHERE user_id = :uid AND code_hash = :h "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"uid": user["id"], "h": code_hash},
        ).first()
        if row is None:
            return False

        token_id, expires_at, used_at = row[0], row[1], row[2]
        if used_at is not None:
            return False
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if now > expires_at:
            return False

        conn.execute(
            text("UPDATE users SET password_hash = :pw WHERE id = :id"),
            {"pw": _hash_password(new_password), "id": user["id"]},
        )
        conn.execute(
            text("UPDATE password_reset_codes SET used_at = :now WHERE id = :id"),
            {"now": now, "id": token_id},
        )
        # Invalidate any other outstanding codes for this user.
        conn.execute(
            text(
                "UPDATE password_reset_codes SET used_at = :now "
                "WHERE user_id = :uid AND used_at IS NULL"
            ),
            {"now": now, "uid": user["id"]},
        )
        return True


# ── Favorites ─────────────────────────────────────────────────────

def get_favorites(user_id: int) -> list[str]:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text("SELECT ticker FROM favorites WHERE user_id = :uid ORDER BY ticker"),
            {"uid": int(user_id)},
        ).all()
    return [r[0] for r in rows]


def add_favorite(user_id: int, ticker: str) -> None:
    ticker = (ticker or "").strip().upper()
    if not ticker or len(ticker) > 10:
        raise ValueError("Ticker must be 1-10 characters.")
    now = datetime.now(timezone.utc)
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO favorites (user_id, ticker, created_at) "
                "VALUES (:uid, :t, :now) "
                "ON CONFLICT (user_id, ticker) DO NOTHING"
            ),
            {"uid": int(user_id), "t": ticker, "now": now},
        )


def remove_favorite(user_id: int, ticker: str) -> None:
    ticker = (ticker or "").strip().upper()
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM favorites WHERE user_id = :uid AND ticker = :t"),
            {"uid": int(user_id), "t": ticker},
        )


def set_favorites(user_id: int, tickers: list[str]) -> None:
    """Replace the user's favorites list with the given list."""
    now = datetime.now(timezone.utc)
    cleaned: list[str] = []
    for t in tickers:
        t = (t or "").strip().upper()
        if t and len(t) <= 10 and t not in cleaned:
            cleaned.append(t)
    with get_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM favorites WHERE user_id = :uid"),
            {"uid": int(user_id)},
        )
        if cleaned:
            conn.execute(
                text(
                    "INSERT INTO favorites (user_id, ticker, created_at) "
                    "VALUES (:uid, :t, :now) "
                    "ON CONFLICT (user_id, ticker) DO NOTHING"
                ),
                [{"uid": int(user_id), "t": t, "now": now} for t in cleaned],
            )


# ── Telegram credentials ──────────────────────────────────────────

def set_telegram_credentials(user_id: int, bot_token: str, chat_id: str) -> None:
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    if not bot_token or ":" not in bot_token or len(bot_token) > 200:
        raise ValueError("Bot token looks invalid. Expected format: 123456789:AA...")
    if not chat_id or len(chat_id) > 64:
        raise ValueError("Chat id is required (the number from getUpdates).")
    cid_test = chat_id[1:] if chat_id.startswith("-") else chat_id
    if not cid_test.isdigit():
        raise ValueError("Chat id must be a number (e.g. 123456789 or -100123).")
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE users SET telegram_bot_token = :tok, "
                "telegram_chat_id = :cid WHERE id = :id"
            ),
            {"tok": bot_token, "cid": chat_id, "id": int(user_id)},
        )


def clear_telegram_credentials(user_id: int) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE users SET telegram_bot_token = NULL, "
                "telegram_chat_id = NULL WHERE id = :id"
            ),
            {"id": int(user_id)},
        )


def get_telegram_credentials(user_id: int) -> tuple[str | None, str | None]:
    with get_engine().begin() as conn:
        row = conn.execute(
            text(
                "SELECT telegram_bot_token, telegram_chat_id "
                "FROM users WHERE id = :id"
            ),
            {"id": int(user_id)},
        ).first()
        if not row:
            return None, None
        return row[0], row[1]
