"""
PDF Processing Module
Extracts text, structure, and metadata from PDF documents
"""
import pdfplumber
import PyPDF2
from typing import Dict, List, Any
from io import BytesIO
import re


class PDFProcessor:
    """Process PDF documents to extract text and structure"""

    def __init__(self):
        pass

    def process(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Process a PDF file and extract all relevant information

        Args:
            file_bytes: Raw PDF file bytes

        Returns:
            Dictionary containing extracted text, metadata, and structure
        """
        file_obj = BytesIO(file_bytes)

        # Extract basic metadata
        metadata = self._extract_metadata(file_obj)

        # Reset file pointer
        file_obj.seek(0)

        # Extract text with structure
        pages = self._extract_text_with_structure(file_obj)

        # Calculate statistics
        total_text = " ".join([p["text"] for p in pages])
        word_count = len(total_text.split())
        char_count = len(total_text)

        return {
            "metadata": metadata,
            "pages": pages,
            "statistics": {
                "page_count": len(pages),
                "word_count": word_count,
                "char_count": char_count,
            },
            "full_text": total_text,
        }

    def _extract_metadata(self, file_obj: BytesIO) -> Dict[str, Any]:
        """Extract PDF metadata using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            metadata = pdf_reader.metadata or {}

            return {
                "title": metadata.get("/Title", ""),
                "author": metadata.get("/Author", ""),
                "subject": metadata.get("/Subject", ""),
                "creator": metadata.get("/Creator", ""),
                "producer": metadata.get("/Producer", ""),
                "creation_date": str(metadata.get("/CreationDate", "")),
                "modification_date": str(metadata.get("/ModDate", "")),
                "page_count": len(pdf_reader.pages),
            }
        except Exception as e:
            return {"error": str(e)}

    def _extract_text_with_structure(self, file_obj: BytesIO) -> List[Dict[str, Any]]:
        """Extract text with structural information using pdfplumber"""
        pages = []

        try:
            with pdfplumber.open(file_obj) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    text = page.extract_text() or ""

                    # Try to detect headings (simple heuristic)
                    headings = self._detect_headings(text)

                    # Extract tables if present
                    tables = page.extract_tables()

                    # Get page dimensions
                    page_info = {
                        "page_number": page_num,
                        "text": text,
                        "headings": headings,
                        "has_tables": len(tables) > 0,
                        "table_count": len(tables),
                        "tables": tables if tables else [],
                        "width": page.width,
                        "height": page.height,
                    }

                    pages.append(page_info)

        except Exception as e:
            # Fallback to PyPDF2 if pdfplumber fails
            pages = self._fallback_extraction(file_obj)

        return pages

    def _fallback_extraction(self, file_obj: BytesIO) -> List[Dict[str, Any]]:
        """Fallback extraction using PyPDF2"""
        file_obj.seek(0)
        pages = []

        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append(
                    {
                        "page_number": page_num,
                        "text": text,
                        "headings": self._detect_headings(text),
                        "has_tables": False,
                        "table_count": 0,
                        "tables": [],
                    }
                )
        except Exception as e:
            raise Exception(f"Failed to extract PDF: {str(e)}")

        return pages

    def _detect_headings(self, text: str) -> List[str]:
        """
        Simple heuristic to detect potential headings
        Looks for short lines (< 80 chars) that start with capital letters
        and don't end with periods
        """
        if not text:
            return []

        lines = text.split("\n")
        headings = []

        for line in lines:
            line = line.strip()
            # Heuristic: short lines, starts with capital, no period at end
            if (
                line
                and len(line) < 80
                and line[0].isupper()
                and not line.endswith(".")
                and not line.endswith(",")
                and len(line.split()) < 12  # Less than 12 words
            ):
                # Check if it looks like a heading (not just a sentence fragment)
                if re.match(r"^[A-Z][A-Za-z0-9\s\-:]+$", line):
                    headings.append(line)

        return headings[:20]  # Limit to 20 headings per page
