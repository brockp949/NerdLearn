import PyPDF2
import os
from app.services.vector_store import VectorStoreService
from app.services.graph_service import AsyncGraphService
from app.services.community_service import CommunityDetectionService
from typing import List, Dict, Any, Optional

class IngestionService:
    """Service for ingesting content files"""

    def __init__(self, db=None):
        self.db = db
        self.vector_store = VectorStoreService(db=self.db)
        self.graph_service = AsyncGraphService(db=self.db)

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
        
        # Graph Construction
        try:
            # Join text for better extraction context
            full_text = "\n".join([d["text"] for d in documents])
            print(f"DEBUG: Processing {len(full_text)} chars for graph extraction.")
            concepts = self.graph_service.extract_concepts(full_text)
            print(f"DEBUG: Extracted {len(concepts)} concepts: {concepts[:5]}...")
            
            # Create generic module for file if not exists
            # We use file basename as module title for now
            module_id = abs(hash(file_path)) % 100000 
            await self.graph_service.create_module_node(course_id, module_id, os.path.basename(file_path))
            
            for concept in concepts:
                await self.graph_service.create_concept_node(
                    course_id=course_id,
                    module_id=module_id,
                    name=concept
                )
            
            await self.db.commit()
            
        except Exception as e:
            print(f"Graph update failed: {e}")
            await self.db.rollback()
            
        return count

    async def optimize_course(self, course_id: int):
        """Run community detection and summarization"""
        cd_service = CommunityDetectionService(db=self.db)
        await cd_service.run_detection(course_id)
        await cd_service.summarize_communities(course_id)

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
