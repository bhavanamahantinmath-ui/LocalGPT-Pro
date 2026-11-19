"""
Document loader and chunking engine for LocalGPT.
Extracts text from PDF, DOCX, and TXT files and splits them into
semantically coherent chunks with source and page attribution.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pypdf
import docx

class DocumentChunk:
    def __init__(self, text: str, source: str, page: int = 1, chunk_index: int = 0):
        self.text = text
        self.source = source
        self.page = page
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index
        }

class DocumentLoader:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str, filename: Optional[str] = None) -> List[DocumentChunk]:
        """
        Loads and splits a document into chunks with metadata.
        """
        filename = filename or os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return self._load_pdf(file_path, filename)
        elif ext in [".docx", ".doc"]:
            return self._load_docx(file_path, filename)
        elif ext in [".txt", ".md", ".csv", ".json", ".py"]:
            return self._load_text(file_path, filename)
        else:
            # Fallback text read
            return self._load_text(file_path, filename)

    def _load_pdf(self, file_path: str, filename: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        try:
            reader = pypdf.PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_text = self._clean_text(page_text)
                if not page_text:
                    continue

                page_chunks = self._split_text(page_text)
                for chunk_str in page_chunks:
                    chunks.append(DocumentChunk(
                        text=chunk_str,
                        source=filename,
                        page=page_idx + 1,
                        chunk_index=chunk_idx
                    ))
                    chunk_idx += 1
        except Exception as e:
            print(f"Error loading PDF {filename}: {e}")
            raise e

        return chunks

    def _load_docx(self, file_path: str, filename: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        try:
            doc = docx.Document(file_path)
            full_text = []

            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())

            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        full_text.append(row_text)

            combined_text = "\n\n".join(full_text)
            combined_text = self._clean_text(combined_text)
            
            # DOCX does not have fixed pages; approximate page every ~2000 chars
            split_chunks = self._split_text(combined_text)
            for idx, c in enumerate(split_chunks):
                approx_page = max(1, (idx * (self.chunk_size - self.chunk_overlap)) // 2000 + 1)
                chunks.append(DocumentChunk(
                    text=c,
                    source=filename,
                    page=approx_page,
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1
        except Exception as e:
            print(f"Error loading DOCX {filename}: {e}")
            raise e

        return chunks

    def _load_text(self, file_path: str, filename: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        try:
            content = ""
            for encoding in ["utf-8", "latin-1", "utf-16"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            content = self._clean_text(content)
            split_chunks = self._split_text(content)
            for idx, c in enumerate(split_chunks):
                approx_page = max(1, (idx * (self.chunk_size - self.chunk_overlap)) // 2000 + 1)
                chunks.append(DocumentChunk(
                    text=c,
                    source=filename,
                    page=approx_page,
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1
        except Exception as e:
            print(f"Error loading text file {filename}: {e}")
            raise e

        return chunks

    def _clean_text(self, text: str) -> str:
        # Normalize whitespace and carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """
        Splits text using recursive sliding window honoring sentence and paragraph breaks.
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Try to break at a paragraph boundary
                para_break = text.rfind("\n\n", start, end)
                if para_break != -1 and para_break > start + self.chunk_size // 3:
                    end = para_break
                else:
                    # Try to break at a sentence boundary (. ! ?)
                    match = re.search(r"[\.\!\?]\s+", text[start + self.chunk_size // 2:end])
                    if match:
                        end = start + self.chunk_size // 2 + match.end()
                    else:
                        # Fallback to last space
                        space_break = text.rfind(" ", start, end)
                        if space_break != -1 and space_break > start + self.chunk_size // 2:
                            end = space_break

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            if end >= text_len:
                break

            # Slide window with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

