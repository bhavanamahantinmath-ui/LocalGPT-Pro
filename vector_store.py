"""
Per-User FAISS Vector Store for LocalGPT Pro.
Ensures complete vector space isolation between users in:
data/users/<uid>/faiss_index/
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

USERS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users")

class VectorStore:
    def __init__(self, uid: str, dimension: int = 384, custom_dir: Optional[str] = None):
        self.uid = uid
        self.dimension = dimension
        
        if custom_dir:
            self.index_dir = custom_dir
        else:
            self.index_dir = os.path.join(USERS_BASE_DIR, uid, "faiss_index")
            
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.meta_path = os.path.join(self.index_dir, "chunks_metadata.json")

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.load()

    def _create_index(self, dim: int):
        self.dimension = dim
        self.index = faiss.IndexFlatIP(self.dimension)

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds text chunks and their embeddings into the user's isolated FAISS index.
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return

        dim = embeddings.shape[1]
        if self.index is None or self.dimension != dim:
            self._create_index(dim)

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Add to FAISS
        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)

        # Persist index
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Searches the user's FAISS index with query embedding.
        """
        if self.index is None or self.index.ntotal == 0 or len(self.chunks_metadata) == 0:
            return []

        if query_embedding.shape[1] != self.dimension:
            return []

        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        k = min(top_k, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            if dist < score_threshold:
                continue

            meta = self.chunks_metadata[idx]
            match_item = {
                "chunk_id": int(idx),
                "text": meta.get("text", ""),
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 1),
                "chunk_index": meta.get("chunk_index", 0),
                "score": float(dist)
            }
            results.append(match_item)

        return results

    def save(self):
        """Persists the FAISS index and metadata to the user's directory."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_metadata, f, indent=2)
        except Exception as e:
            print(f"[VectorStore:{self.uid}] Error saving index: {e}")

    def load(self):
        """Loads index and metadata if files exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.chunks_metadata = json.load(f)
            except Exception as e:
                print(f"[VectorStore:{self.uid}] Failed to load index, starting fresh: {e}")
                self.clear()
        else:
            self._create_index(self.dimension)

    def clear(self):
        """Resets the vector index and metadata for this user."""
        self._create_index(self.dimension)
        self.chunks_metadata = []
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except Exception:
                pass
        if os.path.exists(self.meta_path):
            try:
                os.remove(self.meta_path)
            except Exception:
                pass

    def delete_by_source(self, source_filename: str):
        """
        Removes chunks belonging to a particular document and rebuilds the FAISS index.
        """
        remaining_chunks = [c for c in self.chunks_metadata if c.get("source") != source_filename]
        if len(remaining_chunks) == len(self.chunks_metadata):
            return

        if len(remaining_chunks) == 0:
            self.clear()
            return

        self.chunks_metadata = remaining_chunks
        self.save()

    def get_indexed_files(self) -> List[Dict[str, Any]]:
        summary = {}
        for c in self.chunks_metadata:
            src = c.get("source", "Unknown")
            if src not in summary:
                summary[src] = {"source": src, "chunk_count": 0, "pages": set()}
            summary[src]["chunk_count"] += 1
            summary[src]["pages"].add(c.get("page", 1))

        result = []
        for src, data in summary.items():
            result.append({
                "source": src,
                "chunks": data["chunk_count"],
                "num_pages": len(data["pages"])
            })
        return result

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0

