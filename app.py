"""
LocalGPT Pro — Secure, Multi-User Local AI Assistant with Firebase Authentication.
Run locally on personal hardware with Qwen/Qwen2.5-Instruct.
"""

import os
import sys
import streamlit as st

# Ensure LocalGPT-Pro directory and root are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from database import Database
from ui.theme import inject_theme
from pages.login import render_login_page

# Streamlit Page Config
st.set_page_config(
    page_title="LocalGPT Pro — Private Enterprise AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Cached Model & Shared Core Engines (Loaded on-demand after authentication)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_shared_engine():
    from model import LocalLLMEngine
    engine = LocalLLMEngine()
    engine.load()
    return engine

@st.cache_resource(show_spinner=False)
def get_database():
    return Database()

@st.cache_resource(show_spinner=False)
def get_shared_embedder():
    from embeddings import EmbeddingModel
    return EmbeddingModel()

@st.cache_resource(show_spinner=False)
def get_shared_loader():
    from document_loader import DocumentLoader
    return DocumentLoader()

# Initialize Database Singleton (Instant)
db = get_database()

# Load CSS Stylesheet
css_path = os.path.join(current_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Navigation State
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "workspace"

user = st.session_state.get("user")

# =============================================================================
# ROUTE DISPATCHER
# =============================================================================
if not user:
    # Unauthenticated -> Login Screen (Loads instantly!)
    inject_theme("dark")
    render_login_page(db=db)
else:
    # Authenticated User Session
    from memory import ConversationMemory
    from rag import RAGPipeline
    from chat import ChatCoordinator
    from ui.sidebar import render_sidebar
    from pages.workspace import render_workspace_page
    from pages.documents import render_documents_page
    from pages.profile import render_profile_page

    user_theme = user.get("theme", "dark")
    inject_theme(user_theme)

    # Lazy-load shared model & RAG engines once user signs in
    with st.spinner("⚡ Initializing local Qwen LLM engine & RAG pipeline..."):
        engine = get_shared_engine()
        embedder = get_shared_embedder()
        loader = get_shared_loader()

    uid = user["uid"]

    # Initialize per-user isolated memory and RAG pipeline
    if "memory" not in st.session_state or st.session_state.get("memory_uid") != uid:
        memory = ConversationMemory(uid=uid, db=db)
        memory.initialize_or_resume()
        st.session_state.memory = memory
        st.session_state.memory_uid = uid
    else:
        memory = st.session_state.memory

    if "rag_pipeline" not in st.session_state or st.session_state.get("rag_uid") != uid:
        rag = RAGPipeline(uid=uid, loader=loader, embedder=embedder, db=db)
        st.session_state.rag_pipeline = rag
        st.session_state.rag_uid = uid
    else:
        rag = st.session_state.rag_pipeline

    chat = ChatCoordinator(engine=engine, memory=memory, rag=rag)

    # Render Multi-User Sidebar with Navigation
    active_page = render_sidebar(user=user, db=db, memory=memory)

    # Page Router
    if active_page == "documents":
        render_documents_page(user=user, db=db, rag=rag)
    elif active_page == "profile":
        render_profile_page(user=user, db=db)
    else:
        render_workspace_page(
            user=user,
            db=db,
            memory=memory,
            rag=rag,
            chat=chat,
            engine=engine
        )

