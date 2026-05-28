"""
auth.py
=======
Login gate for the Streamlit app.

Use `require_login()` at the top of every page. It renders a login /
register screen and halts execution if no user is signed in. Otherwise
it returns the current user dict.

Supports:
  - Username + password (bcrypt, stored in data/users.db)
  - Optional Google OAuth via Streamlit's native st.login()
    (requires an [auth] section in .streamlit/secrets.toml)
"""

from __future__ import annotations

import streamlit as st

import auth_db

_AUTH_DB = "data/users.db"
_SESSION_KEY = "auth_user"


# ── Public API ────────────────────────────────────────────────────

def get_current_user() -> dict | None:
    """Return the logged-in user dict from session state, or None."""
    user = st.session_state.get(_SESSION_KEY)
    if not user:
        return None
    # Refresh from DB so any update (e.g. email change) is picked up.
    fresh = auth_db.get_user_by_id(user["id"], db_path=_AUTH_DB)
    if fresh is None:
        st.session_state.pop(_SESSION_KEY, None)
        return None
    st.session_state[_SESSION_KEY] = fresh
    return fresh


def logout() -> None:
    """Clear session and (if used) sign out of Streamlit native auth."""
    st.session_state.pop(_SESSION_KEY, None)
    if _native_login_available():
        try:
            st.logout()
        except Exception:
            pass


def user_db_suffix(user: dict) -> str:
    """Stable, filesystem-safe slug for per-user DB paths."""
    return f"u{int(user['id'])}"


def require_login() -> dict:
    """
    Gate page access. Returns the current user dict when logged in.
    When no user is logged in, renders the login UI and calls st.stop().
    """
    auth_db.init_db(_AUTH_DB)

    user = get_current_user()
    if user is not None:
        return user

    # Bridge Streamlit native auth → our user table.
    if _native_login_available():
        try:
            native = st.user
            if getattr(native, "is_logged_in", False):
                sub = getattr(native, "sub", None) or getattr(native, "email", None)
                email = getattr(native, "email", "") or ""
                name = getattr(native, "name", "") or ""
                if sub:
                    user = auth_db.upsert_google_user(
                        google_sub=str(sub),
                        email=email,
                        name=name,
                        db_path=_AUTH_DB,
                    )
                    st.session_state[_SESSION_KEY] = user
                    st.rerun()
        except Exception:
            pass

    _render_login_page()
    st.stop()


def render_sidebar_user_box() -> None:
    """Show the current user + a logout button in the sidebar."""
    user = get_current_user()
    if not user:
        return
    with st.sidebar:
        st.markdown(f"**👤 {user['username']}**")
        if user.get("email"):
            st.caption(user["email"])
        if st.button("🚪 Log out", use_container_width=True, key="_auth_logout"):
            logout()
            st.rerun()
        st.divider()


# ── Login page ────────────────────────────────────────────────────

def _render_login_page() -> None:
    st.title("🔐 Beat the ASPP — Sign in")
    st.caption("Your favorites, portfolio, and saved data are kept private per account.")

    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            user = auth_db.login_with_password(
                username=username, password=password, db_path=_AUTH_DB
            )
            if user is None:
                st.error("Invalid username or password.")
            else:
                st.session_state[_SESSION_KEY] = user
                st.rerun()

    with tab_register:
        with st.form("register_form", clear_on_submit=False):
            r_username = st.text_input(
                "Username",
                key="reg_username",
                help="3-32 chars · letters, digits, dot, underscore, hyphen",
            )
            r_email = st.text_input("Email (optional)", key="reg_email")
            r_password = st.text_input(
                "Password",
                type="password",
                key="reg_password",
                help="Min 6 characters",
            )
            r_password2 = st.text_input(
                "Confirm password",
                type="password",
                key="reg_password2",
            )
            r_submitted = st.form_submit_button(
                "Create account", type="primary", use_container_width=True
            )
        if r_submitted:
            if r_password != r_password2:
                st.error("Passwords do not match.")
            else:
                try:
                    user = auth_db.register_user(
                        username=r_username,
                        password=r_password,
                        email=r_email,
                        db_path=_AUTH_DB,
                    )
                    st.session_state[_SESSION_KEY] = user
                    st.success(f"Welcome, {user['username']}!")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    if _native_login_available():
        st.divider()
        st.caption("Or sign in with a third-party provider:")
        if st.button("🔵 Continue with Google", use_container_width=True, key="_google_login_btn"):
            try:
                st.login("google")
            except Exception as e:
                st.error(f"Google sign-in failed: {e}")
    else:
        with st.expander("ℹ️ Want to enable Google sign-in?"):
            st.caption(
                "Add an `[auth]` section with a `google` provider to "
                "`.streamlit/secrets.toml` (see Streamlit docs on "
                "`st.login`). Until then, use a username and password."
            )


# ── Internals ─────────────────────────────────────────────────────

def _native_login_available() -> bool:
    """st.login exists (Streamlit ≥ 1.42) AND an [auth] block is configured."""
    if not hasattr(st, "login"):
        return False
    try:
        auth_cfg = st.secrets.get("auth", None)
        return bool(auth_cfg)
    except Exception:
        return False
