"""
Semantic Chunking Module
Intelligently splits documents into semantic chunks for better RAG retrieval
"""
from typing import List, Dict, Any
import re
from transformers import AutoTokenizer


class SemanticChunker:
    """
    Splits text into semantic chunks using a sliding window approach
    with overlap for context preservation
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: Tokenizer model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chunk_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunks with text and metadata
        """
        if not text or not text.strip():
            return []

        # First, try to split by sections/paragraphs
        sections = self._split_into_sections(text)

        chunks = []
        chunk_id = 0

        for section in sections:
            section_text = section["text"]

            # Tokenize the section
            tokens = self.tokenizer.encode(section_text, add_special_tokens=False)

            # If section fits in one chunk, use it as-is
            if len(tokens) <= self.chunk_size:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": section_text,
                        "token_count": len(tokens),
                        "heading": section.get("heading"),
                        "metadata": metadata or {},
                    }
                )
                chunk_id += 1
            else:
                # Split section into overlapping windows
                window_chunks = self._create_overlapping_chunks(
                    section_text, tokens, section.get("heading")
                )

                for window_chunk in window_chunks:
                    window_chunk["chunk_id"] = chunk_id
                    window_chunk["metadata"] = metadata or {}
                    chunks.append(window_chunk)
                    chunk_id += 1

        return chunks

    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sections based on double newlines and headings

        Args:
            text: Text to split

        Returns:
            List of sections with headings
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)

        sections = []
        current_heading = None

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this paragraph looks like a heading
            lines = para.split("\n")
            first_line = lines[0].strip()

            # Simple heuristic: short line, starts with capital, no period at end
            if (
                len(first_line) < 100
                and first_line
                and first_line[0].isupper()
                and not first_line.endswith(".")
                and len(lines) == 1
            ):
                current_heading = first_line
                # If there's more content, add it as a section
                if len(lines) > 1:
                    sections.append(
                        {"heading": current_heading, "text": "\n".join(lines[1:])}
                    )
            else:
                sections.append({"heading": current_heading, "text": para})

        return sections

    def _create_overlapping_chunks(
        self, text: str, tokens: List[int], heading: str = None
    ) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from a long text

        Args:
            text: Text to chunk
            tokens: Pre-tokenized text
            heading: Optional heading for context

        Returns:
            List of overlapping chunks
        """
        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Get chunk window
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Add heading context if available
            if heading and start_idx == 0:
                display_text = f"{heading}\n\n{chunk_text}"
            else:
                display_text = chunk_text

            chunks.append(
                {"text": display_text, "token_count": len(chunk_tokens), "heading": heading}
            )

            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap

            # Prevent infinite loop
            if end_idx >= len(tokens):
                break

        return chunks

    def chunk_document_pages(
        self, pages: List[Dict[str, Any]], doc_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document that's already split into pages

        Args:
            pages: List of pages with text and metadata
            doc_metadata: Document-level metadata

        Returns:
            List of chunks from all pages
        """
        all_chunks = []

        for page in pages:
            page_text = page.get("text", "")
            page_number = page.get("page_number", 0)

            # Create page-specific metadata
            page_metadata = {
                **(doc_metadata or {}),
                "page_number": page_number,
                "headings": page.get("headings", []),
            }

            # Chunk the page text
            page_chunks = self.chunk_text(page_text, page_metadata)

            all_chunks.extend(page_chunks)

        return all_chunks
