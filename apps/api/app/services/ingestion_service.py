import PyPDF2
import os
from app.services.vector_store import VectorStoreService
from typing import List, Dict, Any

class IngestionService:
    """Service for ingesting content files"""

    def __init__(self, db=None):
        self.db = db
        self.vector_store = VectorStoreService()

    async def ingest_file(self, file_path: str, course_id: int = 1) -> int:
        """
        Ingest a file (PDF) into the vector store.
        
        Args:
            file_path: Absolute path to the file
            course_id: Associated course ID
            
        Returns:
            Number of chunks indexed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        chunks = self._extract_and_chunk_pdf(file_path)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk["text"],
                "course_id": course_id,
                "module_type": "pdf",
                "page_number": chunk["page"],
                "metadata": {
                    "source": os.path.basename(file_path),
                    "chunk_index": i
                }
            })
            
        count = await self.vector_store.upsert_documents(documents)
        return count

    def _extract_and_chunk_pdf(self, file_path: str, chunk_size: int = 1000) -> List[Dict]:
        """Extract text from PDF and split into chunks"""
        chunks = []
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                        
                    # Simple chunking by characters for now
                    # In production, use langchain.text_splitter.RecursiveCharacterTextSplitter
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunk_text = text[start:end]
                        chunks.append({
                            "text": chunk_text,
                            "page": page_num + 1
                        })
                        start = end - 100 # Overlap
        except Exception as e:
            print(f"Error extracting PDF {file_path}: {e}")
            raise
            
        return chunks
