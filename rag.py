"""
Per-User RAG (Retrieval-Augmented Generation) Pipeline for LocalGPT Pro.
Guarantees strict per-user document storage and search isolation in:
data/users/<uid>/documents/
"""

import os
import shutil
from typing import List, Dict, Any, Tuple, Optional
from document_loader import DocumentLoader, DocumentChunk
from embeddings import EmbeddingModel
from vector_store import VectorStore
from database import Database

USERS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users")

class RAGPipeline:
    def __init__(
        self,
        uid: str,
        loader: Optional[DocumentLoader] = None,
        embedder: Optional[EmbeddingModel] = None,
        db: Optional[Database] = None
    ):
        self.uid = uid
        self.loader = loader or DocumentLoader()
        self.embedder = embedder or EmbeddingModel()
        self.vector_store = VectorStore(uid=self.uid, dimension=self.embedder.dimension)
        self.db = db or Database()

        self.user_docs_dir = os.path.join(USERS_BASE_DIR, self.uid, "documents")
        os.makedirs(self.user_docs_dir, exist_ok=True)

    def ingest_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Saves, parses, chunks, embeds, and indexes a file into the user's isolated store.
        """
        save_path = os.path.join(self.user_docs_dir, filename)
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        file_size = len(file_bytes)
        file_type = os.path.splitext(filename)[1].lower().replace(".", "")

        # 1. Load and Chunk
        chunks = self.loader.load_file(save_path, filename=filename)
        if not chunks:
            return {
                "success": False,
                "message": f"No extractable text found in '{filename}'.",
                "chunks_count": 0
            }

        # Determine number of pages
        max_page = max(c.page for c in chunks) if chunks else 1

        # 2. Embed
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)

        # 3. Add to User's FAISS Vector Store
        chunk_dicts = [c.to_dict() for c in chunks]
        self.vector_store.add_documents(chunk_dicts, embeddings)

        # 4. Register in User's DB
        self.db.register_document(
            uid=self.uid,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            num_chunks=len(chunks),
            num_pages=max_page
        )

        return {
            "success": True,
            "filename": filename,
            "chunks_count": len(chunks),
            "num_pages": max_page,
            "file_size": file_size,
            "file_type": file_type
        }

    def retrieve_context(self, query: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Searches the user's indexed documents for relevant context.
        """
        if self.vector_store.total_chunks == 0:
            return "", []

        query_emb = self.embedder.embed_query(query)
        matches = self.vector_store.search(query_emb, top_k=top_k)
        if not matches:
            return "", []

        context_parts = []
        sources = []
        seen_citations = set()

        for match in matches:
            src = match["source"]
            page = match["page"]
            text = match["text"]
            score = match["score"]

            context_parts.append(f"[Document: {src} | Page: {page}]\n{text}")

            cite_key = f"{src}_{page}"
            if cite_key not in seen_citations:
                seen_citations.add(cite_key)
                snippet = text.replace("\n", " ")[:140] + ("..." if len(text) > 140 else "")
                sources.append({
                    "source": src,
                    "page": page,
                    "score": round(score, 3),
                    "snippet": snippet
                })

        context_string = "\n\n---\n\n".join(context_parts)
        return context_string, sources

    def build_rag_prompt(self, user_query: str, context_string: str) -> str:
        if not context_string:
            return user_query

        return (
            f"You are a helpful and knowledgeable AI assistant answering questions about uploaded documents.\n\n"
            f"Here is relevant context retrieved from the user's documents:\n"
            f"====================================================\n"
            f"{context_string}\n"
            f"====================================================\n\n"
            f"Please answer the user's question clearly, thoroughly, and accurately based on the context above.\n"
            f"Cite the source document and page number where relevant.\n"
            f"If the information is not contained in the context, say so politely.\n\n"
            f"User Question: {user_query}"
        )

    def delete_document(self, filename: str):
        save_path = os.path.join(self.user_docs_dir, filename)
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        self.vector_store.delete_by_source(filename)
        self.db.delete_document(self.uid, filename)

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        return self.db.get_user_documents(self.uid)

    def clear_all_documents(self):
        self.vector_store.clear()
        if os.path.exists(self.user_docs_dir):
            for f in os.listdir(self.user_docs_dir):
                fpath = os.path.join(self.user_docs_dir, f)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        # Clear from DB
        docs = self.db.get_user_documents(self.uid)
        for d in docs:
            self.db.delete_document(self.uid, d["filename"])

