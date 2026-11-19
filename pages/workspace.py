"""
Main Chat Workspace for LocalGPT Pro.
Features multi-turn streaming, grounded document RAG, export (MD/PDF),
inline collapsible X-Ray drawer, and message branching.
"""

import streamlit as st
from typing import Dict, Any, Optional

from auth.guards import require_login
from database import Database
from memory import ConversationMemory
from chat import ChatCoordinator
from rag import RAGPipeline
from model import LocalLLMEngine
from xray import LLMXRayInspector
from export import export_to_markdown, export_to_pdf
from ui.components import render_welcome_hero

def render_workspace_page(
    user: Dict[str, Any],
    db: Database,
    memory: ConversationMemory,
    rag: RAGPipeline,
    chat: ChatCoordinator,
    engine: LocalLLMEngine
):
    require_login()
    uid = user["uid"]

    # State flags
    if "active_xray_text" not in st.session_state:
        st.session_state["active_xray_text"] = None
    if "editing_msg_id" not in st.session_state:
        st.session_state["editing_msg_id"] = None
    if "copied_msg_id" not in st.session_state:
        st.session_state["copied_msg_id"] = None

    xray_inspector = LLMXRayInspector(engine=engine)
    user_settings = db.get_settings(uid)

    # -------------------------------------------------------------------------
    # Header Bar
    # -------------------------------------------------------------------------
    col_title, col_rag_badge, col_export = st.columns([3, 1.5, 1.5])
    
    with col_title:
        st.subheader(f"💬 {memory.active_title}")

    with col_rag_badge:
        has_docs = rag.vector_store.total_chunks > 0
        enable_rag = st.checkbox(
            f"📄 RAG ({rag.vector_store.total_chunks} chunks)",
            value=has_docs,
            help="Retrieves relevant context from your private uploaded documents."
        )

    with col_export:
        if memory.messages:
            with st.popover("📥 Export Chat", use_container_width=True):
                st.markdown("##### Export Conversation")
                # Markdown export
                md_text = export_to_markdown(memory.active_title, memory.messages)
                st.download_button(
                    label="📄 Download Markdown (.md)",
                    data=md_text,
                    file_name=f"{memory.active_title.replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

                # PDF export
                try:
                    pdf_bytes = export_to_pdf(memory.active_title, memory.messages)
                    st.download_button(
                        label="📑 Download PDF (.pdf)",
                        data=pdf_bytes,
                        file_name=f"{memory.active_title.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.caption(f"PDF generator notice: {e}")

    # -------------------------------------------------------------------------
    # Welcome Hero if Chat is Empty
    # -------------------------------------------------------------------------
    if not memory.messages:
        render_welcome_hero(user.get("name", "User"))

        chip1, chip2, chip3 = st.columns(3)
        with chip1:
            if st.button("✏️ Write or edit", use_container_width=True):
                st.session_state["pending_prompt"] = "Help me write and edit a clear, professional text."
                st.rerun()
        with chip2:
            if st.button("🌐 Search the web", use_container_width=True):
                st.session_state["pending_prompt"] = "Summarize the latest advancements in modern AI and local language models."
                st.rerun()
        with chip3:
            if st.button("🇮🇳 Kannada vocabulary", use_container_width=True):
                st.session_state["pending_prompt"] = "List common Kannada vocabulary words with their English meanings and usage."
                st.rerun()

    # -------------------------------------------------------------------------
    # Message History Stream
    # -------------------------------------------------------------------------
    for idx, message in enumerate(memory.messages):
        role = message["role"]
        content = message["content"]
        msg_id = message["id"]
        sources = message.get("sources")

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                if st.session_state["editing_msg_id"] == msg_id:
                    st.markdown("**Edit Prompt:**")
                    new_val = st.text_area("Update message", value=content, key=f"edit_val_{msg_id}")
                    c_save, c_cancel = st.columns([1, 1])
                    with c_save:
                        if st.button("💾 Save & Re-branch", key=f"save_edit_{msg_id}", type="primary"):
                            st.session_state["editing_msg_id"] = None
                            st.session_state["trigger_edit"] = {"msg_id": msg_id, "new_content": new_val}
                            st.rerun()
                    with c_cancel:
                        if st.button("Cancel", key=f"cancel_edit_{msg_id}"):
                            st.session_state["editing_msg_id"] = None
                            st.rerun()
                else:
                    st.markdown(content)
                    if st.button("✏️ Edit", key=f"edit_btn_{msg_id}", help="Edit this message and re-generate answer"):
                        st.session_state["editing_msg_id"] = msg_id
                        st.rerun()

        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

                # Source references
                if sources:
                    with st.expander(f"📚 Grounded in {len(sources)} Verified Document Source(s)", expanded=False):
                        for s in sources:
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <strong>📄 {s['source']}</strong> — <span style="color: #38bdf8;">Page {s['page']}</span> 
                                    <span style="opacity: 0.6; font-size: 0.75rem;">(Relevance: {s.get('score', 'N/A')})</span><br>
                                    <span style="color: #cbd5e1; font-style: italic;">"...{s['snippet']}..."</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Action row: Copy, Regenerate, Inline X-Ray Drawer
                col_cp, col_rg, col_xr, col_sp = st.columns([1.1, 1.3, 1.5, 4])
                with col_cp:
                    if st.button("📋 Copy", key=f"copy_msg_{msg_id}"):
                        st.session_state["copied_msg_id"] = msg_id
                        st.toast("✅ Copied to clipboard block!")

                with col_rg:
                    if idx == len(memory.messages) - 1:
                        if st.button("🔄 Regenerate", key=f"regen_{msg_id}"):
                            st.session_state["trigger_regen"] = True
                            st.rerun()

                with col_xr:
                    is_open_here = (st.session_state.get("active_xray_msg_id") == msg_id)
                    xray_label = "❌ Close X-Ray" if is_open_here else "🔬 Inline X-Ray"
                    if st.button(xray_label, key=f"xray_{msg_id}", help="Inspect internal tokens, attention, and layers"):
                        if is_open_here:
                            st.session_state["active_xray_msg_id"] = None
                            st.session_state["active_xray_text"] = None
                        else:
                            st.session_state["active_xray_msg_id"] = msg_id
                            st.session_state["active_xray_text"] = content[:300]
                        st.rerun()

                if st.session_state.get("copied_msg_id") == msg_id:
                    st.code(content, language="markdown")

                # Inline Collapsible X-Ray Drawer
                if st.session_state.get("active_xray_msg_id") == msg_id and st.session_state.get("active_xray_text"):
                    with st.container():
                        st.markdown('<div class="xray-drawer">', unsafe_allow_html=True)
                        xray_inspector.render(text_to_inspect=st.session_state["active_xray_text"])
                        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Triggered Actions (Branch Edit & Regenerate)
    # -------------------------------------------------------------------------
    if "trigger_edit" in st.session_state and st.session_state["trigger_edit"]:
        edit_info = st.session_state.pop("trigger_edit")
        msg_id = edit_info["msg_id"]
        new_query = edit_info["new_content"]

        with st.chat_message("assistant", avatar="🤖"):
            stream_gen, sources, augmented_query = chat.edit_user_message_and_respond(
                msg_id=msg_id,
                new_content=new_query,
                enable_rag=enable_rag,
                temperature=user_settings["temperature"],
                top_k=user_settings["top_k"],
                top_p=user_settings["top_p"],
                max_new_tokens=user_settings["max_tokens"]
            )
            full_response = st.write_stream(stream_gen)
            chat.save_assistant_response(response_text=full_response, sources=sources)
        st.rerun()

    if "trigger_regen" in st.session_state and st.session_state["trigger_regen"]:
        st.session_state["trigger_regen"] = False
        with st.chat_message("assistant", avatar="🤖"):
            regen_result = chat.regenerate_last_response(
                enable_rag=enable_rag,
                temperature=user_settings["temperature"],
                top_k=user_settings["top_k"],
                top_p=user_settings["top_p"],
                max_new_tokens=user_settings["max_tokens"]
            )
            if regen_result:
                stream_gen, sources, augmented_query = regen_result
                full_response = st.write_stream(stream_gen)
                chat.save_assistant_response(response_text=full_response, sources=sources)
        st.rerun()

    # -------------------------------------------------------------------------
    # User Input & Streaming
    # -------------------------------------------------------------------------
    prompt_input = st.chat_input("Ask LocalGPT Pro anything or search your private documents...")

    if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
        prompt_input = st.session_state.pop("pending_prompt")

    if prompt_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_input)

        with st.chat_message("assistant", avatar="🤖"):
            stream_gen, sources, augmented_query = chat.generate_response_stream(
                user_input=prompt_input,
                enable_rag=enable_rag,
                temperature=user_settings["temperature"],
                top_k=user_settings["top_k"],
                top_p=user_settings["top_p"],
                max_new_tokens=user_settings["max_tokens"]
            )
            full_response = st.write_stream(stream_gen)

            if sources:
                with st.expander(f"📚 Grounded in {len(sources)} Verified Document Source(s)", expanded=False):
                    for s in sources:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <strong>📄 {s['source']}</strong> — <span style="color: #38bdf8;">Page {s['page']}</span> 
                                <span style="opacity: 0.6; font-size: 0.75rem;">(Relevance: {s.get('score', 'N/A')})</span><br>
                                <span style="color: #cbd5e1; font-style: italic;">"...{s['snippet']}..."</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            chat.save_assistant_response(response_text=full_response, sources=sources)
        st.rerun()

