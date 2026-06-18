"""
auth.py
=======
Login gate for the Streamlit app.

Use `require_login()` at the top of every page. It renders a login /
register screen and halts execution if no user is signed in. Otherwise
it returns the current user dict.

Supports:
  - Username + password (bcrypt, stored in the shared DB)
  - Optional Google OAuth via Streamlit's native st.login()
    (requires an [auth] section in .streamlit/secrets.toml)
"""

from __future__ import annotations

import streamlit as st

import db as _db
import auth_db
from alerts.email_sender import send_email

_SESSION_KEY = "auth_user"
_RESET_STAGE_KEY = "auth_reset_stage"
_RESET_EMAIL_KEY = "auth_reset_email"


# ── Public API ────────────────────────────────────────────────────

def get_current_user() -> dict | None:
    """Return the logged-in user dict from session state, or None."""
    user = st.session_state.get(_SESSION_KEY)
    if not user:
        return None
    # Refresh from DB so any update is picked up.
    fresh = auth_db.get_user_by_id(user["id"])
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


def require_login() -> dict:
    """
    Gate page access. Returns the current user dict when logged in.
    When no user is logged in, renders the login UI and calls st.stop().
    """
    _db.init_all()

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

    tab_login, tab_register, tab_forgot = st.tabs(
        ["Log in", "Create account", "Forgot password"]
    )

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            user = auth_db.login_with_password(
                username=username, password=password,
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
            r_email = st.text_input(
                "Email",
                key="reg_email",
                help="Required so you can reset your password if you forget it",
            )
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
            elif "@" not in (r_email or "") or "." not in (r_email or ""):
                st.error("A valid email is required for password recovery.")
            else:
                try:
                    user = auth_db.register_user(
                        username=r_username,
                        password=r_password,
                        email=r_email,
                    )
                    st.session_state[_SESSION_KEY] = user
                    st.success(f"Welcome, {user['username']}!")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    with tab_forgot:
        _render_forgot_password()

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


# ── Forgot password ───────────────────────────────────────────────

def _render_forgot_password() -> None:
    """Two-step reset: request a code, then submit code + new password.

    Anti-enumeration: we always show the same success message after the
    request step, regardless of whether the email matched a user.
    """
    stage = st.session_state.get(_RESET_STAGE_KEY, "request")

    st.caption(
        "We'll email you a 6-digit code. It expires in 15 minutes "
        "and can only be used once."
    )

    if stage == "request":
        with st.form("forgot_request_form", clear_on_submit=False):
            fp_email = st.text_input(
                "Account email", key="fp_email_req",
                placeholder="you@example.com",
            )
            req_submitted = st.form_submit_button(
                "Send reset code", type="primary", use_container_width=True
            )
        if req_submitted:
            email = (fp_email or "").strip().lower()
            if "@" not in email:
                st.error("Enter the email address on your account.")
            else:
                code = auth_db.create_password_reset_code(email)
                if code:
                    status = send_email(
                        to_addr=email,
                        subject="Your Beat the ASPP password-reset code",
                        body=(
                            f"Hi,\n\n"
                            f"Your password reset code is: {code}\n\n"
                            f"It expires in 15 minutes and can only be used once.\n"
                            f"If you didn't request this, you can ignore this email.\n"
                        ),
                    )
                    if not status.get("sent"):
                        st.error(
                            "Couldn't send the email: "
                            f"{status.get('error', 'unknown error')}"
                        )
                        return
                    if status.get("transport") == "dev_file":
                        st.info(
                            "SMTP is not configured, so the email was saved "
                            f"locally to `{status.get('path')}` and printed "
                            "to the server console. Copy the code from there."
                        )
                # Same success message either way (anti-enumeration).
                st.session_state[_RESET_STAGE_KEY] = "verify"
                st.session_state[_RESET_EMAIL_KEY] = email
                st.success(
                    "If an account exists for that email, a reset code is on "
                    "its way. Enter it below to set a new password."
                )
                st.rerun()

    else:  # stage == "verify"
        st.write(f"Code sent to **{st.session_state.get(_RESET_EMAIL_KEY, '')}**")
        with st.form("forgot_verify_form", clear_on_submit=False):
            fp_code = st.text_input(
                "6-digit code", key="fp_code", max_chars=6,
                placeholder="123456",
            )
            fp_new = st.text_input(
                "New password", type="password", key="fp_new",
                help="Min 6 characters",
            )
            fp_new2 = st.text_input(
                "Confirm new password", type="password", key="fp_new2",
            )
            col1, col2 = st.columns(2)
            with col1:
                verify_submitted = st.form_submit_button(
                    "Reset password", type="primary", use_container_width=True
                )
            with col2:
                back = st.form_submit_button(
                    "Start over", use_container_width=True
                )
        if back:
            st.session_state.pop(_RESET_STAGE_KEY, None)
            st.session_state.pop(_RESET_EMAIL_KEY, None)
            st.rerun()
        if verify_submitted:
            email = st.session_state.get(_RESET_EMAIL_KEY, "")
            if fp_new != fp_new2:
                st.error("Passwords do not match.")
                return
            try:
                ok = auth_db.reset_password_with_code(
                    email=email, code=fp_code, new_password=fp_new,
                )
            except ValueError as e:
                st.error(str(e))
                return
            if not ok:
                st.error("That code is invalid, expired, or already used.")
                return
            st.session_state.pop(_RESET_STAGE_KEY, None)
            st.session_state.pop(_RESET_EMAIL_KEY, None)
            st.success("Password updated! You can log in with your new password.")


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
