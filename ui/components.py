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
    """Time-of-day greeting and welcome hero banner."""
    import datetime
    hr = datetime.datetime.now().hour
    if hr < 12:
        greeting = "Good morning"
    elif hr < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8)); 
                border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 28px 24px; text-align: center; margin: 20px 0;">
        <h1 style="font-size: 2.1rem; margin-bottom: 8px; color: #f8fafc;">
            {greeting}, {user_name} 👋
        </h1>
        <p style="font-size: 1.05rem; color: #94a3b8; max-width: 650px; margin: 0 auto 15px auto;">
            Your secure, private AI assistant powered by <b>Qwen2.5-Instruct</b>.
            Explore your personal documents with RAG or inspect neural weights with LLM X-Ray.
        </p>
    </div>
    """, unsafe_allow_html=True)

