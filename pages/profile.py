"""
User Profile and Preference Settings for LocalGPT Pro.
"""

import streamlit as st
from typing import Dict, Any

from auth.guards import require_login
from auth.auth import logout
from database import Database

def render_profile_page(user: Dict[str, Any], db: Database):
    require_login()
    uid = user["uid"]
    db_user = db.get_user(uid) or user
    settings = db.get_settings(uid)
    stats = db.get_stats(uid)

    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
            👤 User Profile & Workspace Settings
        </h2>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Manage your personal profile, customize your UI theme, and fine-tune your local LLM inference hyperparameters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Identity & Profile Section
    col_u1, col_u2 = st.columns([1, 3])
    with col_u1:
        u_pic = db_user.get("photo_url")
        if u_pic:
            st.markdown(f'<img src="{u_pic}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; border: 2px solid #6366f1;" />', unsafe_allow_html=True)
        else:
            initial = db_user.get("display_name", "U")[0].upper()
            st.markdown(f'<div style="width: 100px; height: 100px; border-radius: 50%; background: #4f46e5; color: white; display: flex; align-items: center; justify-content: center; font-size: 38px; font-weight: 700;">{initial}</div>', unsafe_allow_html=True)

    with col_u2:
        display_name = st.text_input("Display Name", value=db_user.get("display_name", ""))
        st.text_input("Email Address", value=db_user.get("email", ""), disabled=True)
        st.caption(f"Identity Provider: **{db_user.get('provider', 'Firebase')}** &bull; Member since: **{db_user.get('created_at', '')[:10]}**")

    st.markdown("---")

    # 2. Workspace Usage Statistics
    st.markdown("##### 📊 Workspace Analytics")
    c_st1, c_st2, c_st3 = st.columns(3)
    with c_st1:
        st.metric("Total Conversations", stats["chats"])
    with c_st2:
        st.metric("Total Messages Exchanged", stats["messages"])
    with c_st3:
        st.metric("Personal Knowledge Docs", stats["documents"])

    st.markdown("---")

    # 3. UI Theme & System Instructions
    st.markdown("##### 🎨 Interface & Personality")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        current_theme = db_user.get("theme", "dark")
        selected_theme = st.selectbox(
            "Theme Mode",
            ["dark", "light"],
            index=0 if current_theme == "dark" else 1
        )
    with col_t2:
        system_prompt = st.text_area(
            "Default System Instructions",
            value=db_user.get("default_system_prompt") or "You are LocalGPT Pro, an intelligent, helpful, and concise AI assistant running 100% locally on this device.",
            height=90,
            help="Sets the baseline steering instructions for all your conversations."
        )

    st.markdown("---")

    # 4. LLM Generation Hyperparameters
    st.markdown("##### ⚙️ Model Inference Controls (Saved per User)")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=float(settings["temperature"]), step=0.05)
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=int(settings["top_k"]), step=1)
    with col_h2:
        top_p = st.slider("Top-P (Nucleus)", min_value=0.1, max_value=1.0, value=float(settings["top_p"]), step=0.05)
        max_tokens = st.slider("Max Response Tokens", min_value=64, max_value=1024, value=int(settings["max_tokens"]), step=32)

    # Save Settings Button
    if st.button("💾 Save Changes", type="primary", use_container_width=True):
        db.update_user_preferences(
            uid=uid,
            display_name=display_name,
            theme=selected_theme,
            default_system_prompt=system_prompt
        )
        db.update_settings(
            uid=uid,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens
        )
        user["name"] = display_name
        user["theme"] = selected_theme
        st.session_state["user"] = user
        st.toast("✅ Profile and preferences successfully saved!")
        st.rerun()

    st.markdown("---")

    # Danger Zone
    with st.expander("⚠️ Account Actions"):
        col_dz1, col_dz2 = st.columns([1, 1])
        with col_dz1:
            if st.button("🚪 Log Out of Session", use_container_width=True, type="secondary"):
                logout()
        with col_dz2:
            if st.button("🗑️ Delete All My Chats", use_container_width=True, type="primary"):
                chats = db.get_user_chats(uid)
                for c in chats:
                    db.delete_chat(uid, c["chat_id"])
                st.toast("All chats deleted.")
                st.rerun()

