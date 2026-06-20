"""
db.py
=====
Single SQLAlchemy engine for the whole app.

If DATABASE_URL is set (e.g. via Streamlit Cloud secrets), it points to
Postgres. Otherwise we fall back to SQLite at data/local.db for offline
local dev. Same code paths either way.

URL normalisation:
  - postgres://...                  -> postgresql+psycopg://...
  - postgresql://...                -> postgresql+psycopg://...
  (so plain Neon / Supabase / Render URLs work without editing.)
"""

from __future__ import annotations

import os
import threading

from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import Engine

# Streamlit secrets may not be present (e.g. running CLI tools). Lazy import.
def _get_database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    try:
        import streamlit as st  # type: ignore
        val = st.secrets.get("DATABASE_URL", None)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return None


def _normalise_url(url: str) -> str:
    # SQLAlchemy 2.x requires "postgresql+driver://...". Many providers
    # (Neon, Supabase, Render) hand out plain "postgres://" URLs.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine. Lazy + thread-safe."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine

        raw = _get_database_url()
        if raw:
            url = _normalise_url(raw)
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=5,
                max_overflow=5,
            )
        else:
            # Local dev fallback. One file for everything; cleaner than the
            # old per-user-file scheme since user_id is now an explicit column.
            os.makedirs("data", exist_ok=True)
            _engine = create_engine(
                "sqlite:///data/local.db",
                connect_args={"check_same_thread": False},
            )
        return _engine


def is_postgres() -> bool:
    return get_engine().dialect.name == "postgresql"


# Single shared MetaData — every *_db module registers its tables here so
# init_db() can do one create_all().
metadata = MetaData()


def init_all() -> None:
    """Create any missing tables on the configured engine."""
    # Importing the modules registers their tables on `metadata`.
    import auth_db        # noqa: F401
    import portfolio_db   # noqa: F401
    import portfolio_baseline_db  # noqa: F401
    import youtube_db     # noqa: F401
    metadata.create_all(get_engine())
