"""
Shared-password gate with an automatic expiry.

Env vars:
  COURSE_PASSWORD   the password you announce in class (required)
  ACCESS_UNTIL      ISO timestamp, e.g. 2026-11-05T18:00:00+01:00 (optional);
                    after it the app locks itself for everyone
  ALLOW_NO_PASSWORD set to 1 for local development only

Fails closed: with no COURSE_PASSWORD set, nobody gets in.
"""

import hmac
import os
from datetime import datetime, timezone

import streamlit as st

MAX_ATTEMPTS = 8


def _access_until() -> datetime | None:
    raw = os.getenv("ACCESS_UNTIL", "").strip()
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _locked_screen(title: str, body: str) -> None:
    st.markdown(f"## {title}")
    st.info(body)
    st.stop()


def is_authenticated() -> bool:
    return os.getenv("ALLOW_NO_PASSWORD") == "1" or st.session_state.get("authenticated", False)


def require_access() -> None:
    """Call at the top of the app. Renders the login form and stops the script until access is granted."""
    until = _access_until()
    if until and datetime.now(timezone.utc) > until:
        _locked_screen(
            "This lab is closed",
            "The live session is over. Thanks for joining! "
            "All notebooks remain available on GitHub and in Colab.",
        )

    if os.getenv("ALLOW_NO_PASSWORD") == "1":
        return

    password = os.getenv("COURSE_PASSWORD", "")
    if not password:
        _locked_screen("Not configured", "The instructor has not set a course password yet.")

    if st.session_state.get("authenticated"):
        return

    attempts = st.session_state.get("login_attempts", 0)
    if attempts >= MAX_ATTEMPTS:
        _locked_screen("Too many attempts", "Reload the page and try again.")

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("## Self-Improving AI Agents · Lab")
        st.caption("Enter the password shared in the live session.")
        with st.form("login"):
            entered = st.text_input("Course password", type="password")
            submitted = st.form_submit_button("Enter lab", type="primary", width="stretch")
        if submitted:
            if hmac.compare_digest(entered.encode(), password.encode()):
                st.session_state.authenticated = True
                st.rerun()
            st.session_state.login_attempts = attempts + 1
            st.error("That's not it. Check the chat for the password.")
    st.stop()
