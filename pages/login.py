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
    if "show_auth_modal" not in st.session_state:
        st.session_state["show_auth_modal"] = False

    # Check for authentication payload from component if active
    if st.session_state["show_auth_modal"]:
        render_auth_modal_view(db)
    else:
        render_hero_landing_view(db)

def render_hero_landing_view(db: Database):
    """
    Renders the exact SaaS Hero Landing Page requested.
    """
    # Top subtle navbar
    nav_col1, nav_col2 = st.columns([6, 1.5])
    with nav_col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; padding-top: 10px;">
            <span style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                <span style="color: #2dd4bf;">⚡</span> LocalGPT
            </span>
        </div>
        """, unsafe_allow_html=True)
    with nav_col2:
        if st.button("Sign in", key="nav_signin_btn", help="Sign in to your account with Google or Email"):
            st.session_state["show_auth_modal"] = True
            st.rerun()

    # Hero Main Content
    st.markdown("""
    <div class="hero-container">
        <!-- Top Pill Tag -->
        <div class="hero-badge-wrapper">
            <div class="hero-badge">
                Hosted inference &middot; no GPU, no install
            </div>
        </div>

        <!-- Main Headline -->
        <h1 class="hero-title">
            A ChatGPT-style assistant<br>
            that <span class="gradient-text">knows your documents</span>
        </h1>

        <!-- Subtitle -->
        <p class="hero-subtitle">
            Multi-turn conversations with memory, real-time streaming, markdown and code rendering, 
            and retrieval-augmented answers over your own files &mdash; all in the browser.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Centered CTA Buttons
    _, col_btn1, col_btn2, _ = st.columns([2, 1.6, 1.6, 2])

    with col_btn1:
        st.markdown('<div class="landing-btn-primary">', unsafe_allow_html=True)
        if st.button("Start chatting &rarr;", key="hero_start_chatting_btn", use_container_width=True):
            # Instant 1-click access to demo workspace
            dev_payload = {
                "token": "dev_alice@company.com",
                "uid": "uid_alice",
                "email": "alice@company.com",
                "name": "Alice",
                "picture": "",
                "email_verified": True,
                "provider": "guest"
            }
            user = handle_login(dev_payload, db=db)
            if user:
                st.session_state["user"] = user
                st.session_state["nav_page"] = "workspace"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_btn2:
        st.markdown('<div class="landing-btn-secondary">', unsafe_allow_html=True)
        if st.button("Create an account", key="hero_create_account_btn", use_container_width=True):
            st.session_state["show_auth_modal"] = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Cards Grid (Exact 2-Card layout from screenshot)
    st.markdown("""
    <div class="hero-cards-grid">
        <!-- Card 1 -->
        <div class="hero-feature-card">
            <div class="card-icon">⚡</div>
            <div class="card-title">Streaming answers</div>
            <div class="card-desc">
                Hosted inference over the internet &mdash; replies stream in token by token, on any device.
            </div>
        </div>
        <!-- Card 2 -->
        <div class="hero-feature-card">
            <div class="card-icon">💬</div>
            <div class="card-title">Persistent history</div>
            <div class="card-desc">
                Every conversation is saved to your account. Rename, search, revisit or delete anytime.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_auth_modal_view(db: Database):
    """
    Renders the Firebase & Email authentication portal with a back button to the hero.
    """
    # Back button to return to hero view
    col_back, _ = st.columns([2, 8])
    with col_back:
        if st.button("&larr; Back to overview", key="back_to_hero_btn"):
            st.session_state["show_auth_modal"] = False
            st.rerun()

    st.markdown("""
    <div style="text-align: center; margin: 15px 0 25px 0;">
        <h2 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
            Sign in to <span style="color: #2dd4bf;">LocalGPT</span>
        </h2>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Continue with your Google account, email, or select an offline demo workspace.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_center, col_r = st.columns([1, 1.8, 1])

    with col_center:
        # Render Firebase Web Component (Google Sign-In, Email/Password)
        auth_payload = render_auth_component(key="firebase_login_comp")

        if auth_payload:
            user = handle_login(auth_payload, db=db)
            if user:
                st.session_state["user"] = user
                st.session_state["nav_page"] = "workspace"
                st.session_state["show_auth_modal"] = False
                st.toast(f"👋 Welcome, {user['name']}!")
                st.rerun()

        # Offline / Dev quick-login helper
        st.markdown("---")
        with st.expander("🛠️ Local Offline Developer Accounts", expanded=True):
            st.caption("Sign in with pre-configured isolated test accounts:")
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
                    st.session_state["show_auth_modal"] = False
                    st.rerun()
