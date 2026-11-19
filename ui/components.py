"""
Reusable UI component renderers for LocalGPT Pro.
"""

import streamlit as st
from typing import Optional, Dict, Any

def render_avatar(name: str, picture: Optional[str] = None, is_ai: bool = False) -> str:
    """Returns HTML for user avatar or AI badge."""
    if is_ai:
        return (
            '<div style="width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, #4f46e5, #06b6d4); '
            'display: flex; align-items: center; justify-content: center; font-size: 16px; color: white; font-weight: bold; flex-shrink: 0;">'
            '🤖</div>'
        )

    if picture:
        return (
            f'<img src="{picture}" style="width: 34px; height: 34px; border-radius: 50%; object-fit: cover; border: 1px solid #4f46e5; flex-shrink: 0;" />'
        )

    # Initials
    initials = "".join([part[0].upper() for part in name.split()[:2]]) if name else "U"
    return (
        f'<div style="width: 34px; height: 34px; border-radius: 50%; background: #374151; '
        f'display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 600; color: #f8fafc; flex-shrink: 0;">'
        f'{initials}</div>'
    )

def render_skeleton():
    """Renders a pulsing loading skeleton during generation."""
    skeleton_html = """
    <div style="padding: 14px; animation: pulse 1.5s infinite ease-in-out;">
        <div style="height: 14px; background: rgba(148, 163, 184, 0.2); border-radius: 4px; width: 65%; margin-bottom: 8px;"></div>
        <div style="height: 14px; background: rgba(148, 163, 184, 0.2); border-radius: 4px; width: 85%; margin-bottom: 8px;"></div>
        <div style="height: 14px; background: rgba(148, 163, 184, 0.2); border-radius: 4px; width: 40%;"></div>
    </div>
    <style>
        @keyframes pulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 0.8; }
        }
    </style>
    """
    return st.markdown(skeleton_html, unsafe_allow_html=True)

def render_welcome_hero(user_name: str):
    """Headline matching ChatGPT Cloud aesthetics."""
    st.markdown("""
    <div style="text-align: center; margin: 45px auto 30px auto; max-width: 780px;">
        <h1 style="font-size: 3rem; font-weight: 800; letter-spacing: -1.2px; margin-bottom: 8px;">
            How can I help you <span style="color: #0d9488;">today?</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

