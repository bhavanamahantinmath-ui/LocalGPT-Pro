"""
Dynamic Dark/Light theme system for LocalGPT Pro.
"""

import streamlit as st

THEMES = {
    "dark": {
        "bg_main": "#0b0f19",
        "bg_card": "#111827",
        "bg_sidebar": "#0f172a",
        "border": "#1e293b",
        "text_primary": "#f8fafc",
        "text_secondary": "#94a3b8",
        "accent": "#6366f1",
        "accent_gradient": "linear-gradient(135deg, #4f46e5, #6366f1)",
        "user_bubble": "rgba(30, 41, 59, 0.8)",
        "ai_bubble": "rgba(17, 24, 39, 0.7)",
        "code_bg": "#090d16"
    },
    "light": {
        "bg_main": "#f8fafc",
        "bg_card": "#ffffff",
        "bg_sidebar": "#f1f5f9",
        "border": "#e2e8f0",
        "text_primary": "#0f172a",
        "text_secondary": "#64748b",
        "accent": "#4f46e5",
        "accent_gradient": "linear-gradient(135deg, #4338ca, #4f46e5)",
        "user_bubble": "rgba(238, 242, 255, 0.9)",
        "ai_bubble": "rgba(255, 255, 255, 0.95)",
        "code_bg": "#f1f5f9"
    }
}

def inject_theme(theme_name: str = "dark"):
    """
    Injects custom CSS adhering to the selected theme mode.
    """
    t = THEMES.get(theme_name, THEMES["dark"])
    css = f"""
    <style>
        :root {{
            --bg-main: {t['bg_main']};
            --bg-card: {t['bg_card']};
            --bg-sidebar: {t['bg_sidebar']};
            --border-color: {t['border']};
            --text-primary: {t['text_primary']};
            --text-secondary: {t['text_secondary']};
            --accent-color: {t['accent']};
            --accent-grad: {t['accent_gradient']};
            --user-bubble: {t['user_bubble']};
            --ai-bubble: {t['ai_bubble']};
            --code-bg: {t['code_bg']};
        }}

        /* Hide Streamlit Header & Footer */
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
        #MainMenu, footer, .stDeployButton {{
            visibility: hidden;
            display: none !important;
        }}

        /* App Background */
        .stApp {{
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-color) !important;
        }}

        /* Input styling */
        .stTextInput>div>div>input, .stTextArea textarea {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }}

        /* Primary Button */
        button[data-testid="baseButton-primary"] {{
            background: var(--accent-grad) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }}
        button[data-testid="baseButton-primary"]:hover {{
            opacity: 0.92 !important;
            transform: translateY(-1px) !important;
        }}

        /* Secondary Button */
        button[data-testid="baseButton-secondary"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
        }}

        /* Source Card */
        .citation-box {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-left: 3px solid var(--accent-color);
            border-radius: 8px;
            padding: 10px 14px;
            margin: 6px 0;
            font-size: 0.85rem;
        }}

        /* User Profile Pill in Sidebar */
        .user-profile-pill {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-top: 15px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

