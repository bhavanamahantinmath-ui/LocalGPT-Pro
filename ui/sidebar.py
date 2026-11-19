"""
Sidebar layout and navigation component for LocalGPT Pro.
Features chat search, pinning, date grouping, document navigation, and user profile drawer.
"""

import streamlit as st
from typing import Dict, Any, Optional

from database import Database
from memory import ConversationMemory
from auth.auth import logout

def render_sidebar(
    user: Dict[str, Any],
    db: Database,
    memory: ConversationMemory
) -> str:
    """
    Renders the rich multi-user sidebar and returns the active page selection.
    """
    uid = user["uid"]
    active_page = st.session_state.get("nav_page", "workspace")

    with st.sidebar:
        # App Branding Header
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, #4f46e5, #06b6d4); 
                        display: flex; align-items: center; justify-content: center; font-size: 18px; color: white;">
                ⚡
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.5px;">LocalGPT Pro</h3>
                <span style="font-size: 0.72rem; color: #818cf8; font-weight: 600;">ENTERPRISE LOCAL AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Primary '+ New Chat' Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            memory.create_new_chat("New Chat")
            st.session_state["nav_page"] = "workspace"
            st.session_state["active_xray_text"] = None
            st.session_state["editing_msg_id"] = None
            st.rerun()

        # Navigation Options Menu
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            if st.button("💬 Chat", use_container_width=True, type="secondary" if active_page != "workspace" else "primary"):
                st.session_state["nav_page"] = "workspace"
                st.rerun()
        with nav_col2:
            if st.button("📚 Docs", use_container_width=True, type="secondary" if active_page != "documents" else "primary"):
                st.session_state["nav_page"] = "documents"
                st.rerun()
        with nav_col3:
            if st.button("⚙️ Profile", use_container_width=True, type="secondary" if active_page != "profile" else "primary"):
                st.session_state["nav_page"] = "profile"
                st.rerun()

        st.markdown("---")

        # Chat Search Filter
        search_query = st.text_input(
            "Search",
            placeholder="🔍 Search chat history...",
            label_visibility="collapsed",
            key="chat_search_query"
        )

        # Retrieve Date-Grouped Chats for this User
        chat_groups = db.get_user_chats_grouped(uid, search_query=search_query)

        chat_list_container = st.container(height=320)
        with chat_list_container:
            # 1. Pinned Chats
            if chat_groups["Pinned"]:
                st.markdown("<p style='font-size: 11px; font-weight: 700; color: #818cf8; margin: 4px 0;'>📌 PINNED</p>", unsafe_allow_html=True)
                _render_chat_group_items(chat_groups["Pinned"], memory, db, uid)

            # 2. Date Groups
            for group_name in ["Today", "Yesterday", "Previous 7 Days", "Older"]:
                items = chat_groups[group_name]
                if items:
                    st.markdown(f"<p style='font-size: 11px; font-weight: 700; color: #64748b; margin: 8px 0 4px 0;'>{group_name.upper()}</p>", unsafe_allow_html=True)
                    _render_chat_group_items(items, memory, db, uid)

            if not any(chat_groups.values()):
                st.caption("No chats found. Ask anything to start!")

        st.markdown("---")

        # Bottom User Profile Card with Menu
        u_name = user.get("name", "User")
        u_email = user.get("email", "")
        u_pic = user.get("picture", "")

        col_pic, col_info, col_menu = st.columns([1, 3, 1])
        with col_pic:
            if u_pic:
                st.markdown(f'<img src="{u_pic}" style="width: 32px; height: 32px; border-radius: 50%; object-fit: cover;" />', unsafe_allow_html=True)
            else:
                initial = u_name[0].upper() if u_name else "U"
                st.markdown(f'<div style="width: 32px; height: 32px; border-radius: 50%; background: #4f46e5; color: white; display: flex; align-items: center; justify-content: center; font-weight: 600;">{initial}</div>', unsafe_allow_html=True)

        with col_info:
            st.markdown(f"<p style='margin: 0; font-size: 13px; font-weight: 600; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>{u_name}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='margin: 0; font-size: 11px; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>{u_email}</p>", unsafe_allow_html=True)

        with col_menu:
            with st.popover("⋯", help="Account Menu"):
                st.markdown(f"**{u_name}**")
                st.caption(u_email)
                if st.button("👤 View Profile", use_container_width=True, key="popover_profile"):
                    st.session_state["nav_page"] = "profile"
                    st.rerun()
                if st.button("📁 Document Library", use_container_width=True, key="popover_docs"):
                    st.session_state["nav_page"] = "documents"
                    st.rerun()
                if st.button("🚪 Log Out", use_container_width=True, type="primary", key="popover_logout"):
                    logout()

    return st.session_state.get("nav_page", "workspace")

def _render_chat_group_items(items, memory: ConversationMemory, db: Database, uid: str):
    for c in items:
        cid = c["chat_id"]
        title = c["title"]
        is_active = (cid == memory.active_chat_id)
        is_pinned = c["pinned"]

        col_t, col_a = st.columns([4, 1])
        with col_t:
            prefix = "👉 " if is_active else ("📌 " if is_pinned else "💬 ")
            trunc_title = (title[:20] + "...") if len(title) > 20 else title
            if st.button(
                f"{prefix}{trunc_title}",
                key=f"side_chat_{cid}",
                use_container_width=True,
                help=title
            ):
                memory.load_chat(cid)
                st.session_state["nav_page"] = "workspace"
                st.session_state["active_xray_text"] = None
                st.session_state["editing_msg_id"] = None
                st.rerun()

        with col_a:
            with st.popover("⚙️", help="Chat Actions"):
                st.markdown(f"**{title}**")
                new_t = st.text_input("Rename Chat", value=title, key=f"ren_input_{cid}")
                if st.button("Save Title", key=f"save_ren_{cid}"):
                    memory.rename_chat(cid, new_t)
                    st.rerun()

                pin_label = "Unpin Chat" if is_pinned else "Pin Chat"
                if st.button(pin_label, key=f"pin_btn_{cid}"):
                    db.toggle_chat_pin(uid, cid)
                    st.rerun()

                if st.button("🗑️ Delete", key=f"del_chat_{cid}", type="primary"):
                    memory.delete_chat(cid)
                    memory.initialize_or_resume()
                    st.rerun()

