"""
Document Manager and Private Knowledge Base for LocalGPT Pro.
Provides drag-and-drop ingestion, progress feedback, chunk inspection, and index maintenance.
"""

import streamlit as st
from typing import Dict, Any

from auth.guards import require_login
from database import Database
from rag import RAGPipeline

def render_documents_page(user: Dict[str, Any], db: Database, rag: RAGPipeline):
    require_login()
    uid = user["uid"]

    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
            📁 Private Knowledge Base & Document Manager
        </h2>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Upload PDF, Word, or Markdown files to ground your local AI in your personal documents.
            All files are stored in your private encrypted workspace and never shared across users.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Ingestion Section with Progress
    st.markdown("##### 📤 Drag & Drop Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF research papers, Word docs, notes, or code to index into FAISS."
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            total = len(uploaded_files)
            for idx, up_file in enumerate(uploaded_files):
                status_text.text(f"Processing ({idx+1}/{total}): '{up_file.name}'...")
                file_bytes = up_file.read()
                res = rag.ingest_file(file_bytes=file_bytes, filename=up_file.name)
                
                if res["success"]:
                    st.toast(f"✅ Indexed '{up_file.name}' ({res['chunks_count']} chunks)")
                else:
                    st.error(res["message"])

                progress_bar.progress((idx + 1) / total)

            status_text.text("🎉 All documents successfully processed and indexed into FAISS!")
            st.rerun()

    st.markdown("---")

    # 2. Document Library Cards & Management
    docs = rag.get_indexed_documents()
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"##### 📚 Your Indexed Library ({len(docs)} documents • {rag.vector_store.total_chunks} total chunks)")
    with col_h2:
        if docs and st.button("🗑️ Clear Entire Library", type="secondary", use_container_width=True):
            rag.clear_all_documents()
            st.toast("Cleared user library.")
            st.rerun()

    if not docs:
        st.info("You haven't uploaded any documents yet. Drag and drop your first file above to get started!")
    else:
        for d in docs:
            fname = d["filename"]
            ftype = d["file_type"].upper()
            size_kb = round(d["file_size"] / 1024, 1)
            chunks = d["chunks"]
            pages = d.get("pages", 1)
            uploaded_at = d["uploaded_at"][:10]

            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(f"""
                <div class="doc-card">
                    <div>
                        <strong style="color: #f8fafc; font-size: 1rem;">📄 {fname}</strong>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                            <span style="background: rgba(99,102,241,0.2); color: #818cf8; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{ftype}</span>
                            &bull; {size_kb} KB &bull; {pages} page(s) &bull; <b style="color: #38bdf8;">{chunks} chunks</b> &bull; Added {uploaded_at}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_del:
                if st.button("🗑️ Delete", key=f"del_doc_lib_{fname}", use_container_width=True):
                    rag.delete_document(fname)
                    st.toast(f"Removed '{fname}'.")
                    st.rerun()

        # Chunk Inspector Expander
        with st.expander("🔍 Inspect Internal Vector Chunks"):
            raw_meta = rag.vector_store.chunks_metadata
            if raw_meta:
                st.caption(f"Showing sample chunks from your active index ({len(raw_meta)} total):")
                for i, c in enumerate(raw_meta[:10]):
                    st.markdown(f"**Chunk #{i+1}** &bull; Source: `{c.get('source')}` (Page {c.get('page')})")
                    st.code(c.get("text", "")[:250] + "...", language="text")
            else:
                st.caption("No chunks in index.")

