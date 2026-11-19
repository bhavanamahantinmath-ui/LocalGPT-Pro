"""
Route Guard and Access Control for LocalGPT Pro.
Ensures protected pages require a verified active session.
"""

import streamlit as st

def require_login(check_email_verified: bool = True) -> dict:
    """
    Checks active user session.
    If unauthenticated, redirects to the login screen.
    If authenticated via email/password but unverified, enforces verification screen.
    """
    user = st.session_state.get("user")
    if not user:
        st.session_state["nav_page"] = "login"
        st.rerun()
        st.stop()

    if check_email_verified and not user.get("email_verified", True) and user.get("provider") == "password":
        st.warning("✉️ Please verify your email address to access LocalGPT Pro.")
        st.info(f"A verification link was sent to **{user.get('email')}**.")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 I Have Verified (Refresh)", use_container_width=True):
                # Update status
                user["email_verified"] = True
                st.session_state["user"] = user
                st.rerun()
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                from auth.auth import logout
                logout()
        st.stop()

    return user

