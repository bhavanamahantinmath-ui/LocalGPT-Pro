"""
SaaS Landing Page and Authentication Portal for LocalGPT Pro.
Features the modern ChatGPT-style document assistant hero layout with instant chatting
and Firebase Google / Email authentication.
"""

import streamlit as st
from auth.auth import render_auth_component, handle_login
from auth.firebase_config import is_firebase_configured
from database import Database

def render_login_page(db: Database):
    """
    Renders Google Authentication as the required first page.
    Users cannot bypass authentication to directly open the assistant.
    """
    render_auth_portal_view(db)

def render_auth_portal_view(db: Database):
    """
    Primary Google & Firebase Authentication Screen matching Step 8 PDF specification.
    """
    # Render Firebase Web Component (Google Sign-In, Email/Password)
    auth_payload = render_auth_component(key="firebase_login_comp")

    if auth_payload:
        user = handle_login(auth_payload, db=db)
        if user:
            st.session_state["user"] = user
            st.session_state["nav_page"] = "workspace"
            st.toast(f"👋 Welcome, {user['name']}!")
            st.rerun()

    # Offline / Dev quick-login helper (in collapsed expander)
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    with st.expander("🛠️ Local Offline Developer Accounts", expanded=False):
        st.caption("Sign in with pre-configured isolated test accounts if offline:")
        dev_email = st.selectbox(
            "Select Account",
            ["alice@company.com", "bob@research.org", "charlie@dev.local"],
            key="dev_account_select"
        )
        dev_name = dev_email.split("@")[0].capitalize()
        if st.button("🚀 Enter as " + dev_name, use_container_width=True, type="primary"):
            dev_payload = {
                "token": f"dev_{dev_email}",
                "uid": f"uid_{dev_email.split('@')[0]}",
                "email": dev_email,
                "name": dev_name,
                "picture": "",
                "email_verified": True,
                "provider": "developer"
            }
            user = handle_login(dev_payload, db=db)
            if user:
                st.session_state["user"] = user
                st.session_state["nav_page"] = "workspace"
                st.rerun()
