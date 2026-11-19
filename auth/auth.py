"""
Authentication Controller for LocalGPT Pro.
Provides Streamlit bidirectional component binding, token verification,
session lifecycle management, and profile synchronization.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional, Dict, Any

from auth.firebase_config import get_firebase_web_config, is_firebase_configured
from auth.firebase_admin_init import verify_token
from database import Database

COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "component")
_auth_component = components.declare_component("firebase_auth", path=COMPONENT_DIR)

def render_auth_component(key: str = "firebase_auth_comp") -> Optional[Dict[str, Any]]:
    """
    Renders the Firebase Auth bidirectional component.
    Returns authenticated user data or None.
    """
    config = get_firebase_web_config()
    res = _auth_component(firebaseConfig=config, default=None, key=key)
    return res

def handle_login(token_or_payload: Any, db: Database) -> Optional[Dict[str, Any]]:
    """
    Verifies ID token and logs the user into session state.
    """
    if not token_or_payload:
        return None

    # Handle dictionary payload from component
    if isinstance(token_or_payload, dict):
        token = token_or_payload.get("token")
        client_uid = token_or_payload.get("uid")
        email = token_or_payload.get("email")
        name = token_or_payload.get("name")
        pic = token_or_payload.get("picture", "")
        verified = token_or_payload.get("email_verified", True)
        provider = token_or_payload.get("provider", "firebase")
    else:
        token = str(token_or_payload)
        client_uid = None
        email = None
        name = None
        pic = ""
        verified = True
        provider = "firebase"

    try:
        # Backend verification
        verified_data = verify_token(token)
        uid = client_uid or verified_data.get("uid") or f"user_{hash(email)}"
        user_email = email or verified_data.get("email") or f"user_{str(uid)[:6]}@localgpt.pro"
        user_name = name or verified_data.get("name") or user_email.split("@")[0]
        user_pic = pic or verified_data.get("picture") or ""
        is_verified = verified if verified is not None else verified_data.get("email_verified", True)

        # Upsert user record into SQLite
        db_user = db.upsert_user(
            uid=uid,
            email=user_email,
            display_name=user_name,
            photo_url=user_pic,
            provider=provider
        )

        user_session = {
            "uid": uid,
            "email": user_email,
            "name": db_user.get("display_name", user_name),
            "picture": db_user.get("photo_url", user_pic),
            "email_verified": is_verified,
            "provider": provider,
            "theme": db_user.get("theme", "dark"),
            "token": token
        }

        st.session_state["user"] = user_session
        return user_session

    except Exception as e:
        print(f"[Auth] Token verification failed: {e}")
        st.error(f"Login verification failed: {e}")
        return None

def logout():
    """Clears user session and logs out."""
    for key in ["user", "memory", "chat_coordinator", "rag_pipeline"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["nav_page"] = "login"
    st.rerun()

def get_current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("user")

