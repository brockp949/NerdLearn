"""
Vector Store Service for API
Manages similarity search using pgvector
"""
from typing import List, Dict, Any, Optional
import openai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.models.vector_store import CourseChunk
import uuid

class VectorStoreService:
    """Manages vector embeddings in PostgreSQL (pgvector) for the API"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.vector_size = settings.VECTOR_SIZE
        self.embedding_model = settings.EMBEDDING_MODEL
        openai.api_key = settings.OPENAI_API_KEY

    def embed_text(self, text: str) -> List[float]:
        """Wrapper for single text embedding"""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using OpenAI"""
        if not texts:
            return []

        # Return mock embedding if no API key
        if not settings.OPENAI_API_KEY or "your-openai-api-key" in settings.OPENAI_API_KEY:
            print("WARNING: No OpenAI API key found. Using mock embedding.")
            return [[0.0] * self.vector_size for _ in texts]

        try:
            # Efficient batch call
            response = openai.embeddings.create(input=texts, model=self.embedding_model)
            # Ensure order is preserved
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"WARNING: Embedding failed: {e}. Using mock embedding.")
            return [[0.0] * self.vector_size for _ in texts]

    async def search(
        self,
        query: str,
        course_id: int = None,
        module_id: int = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in pgvector"""
        # Generate query embedding
        query_vector = self.embed_text(query)

        # Build query
        # Use l2_distance or cosine_distance (<=>)
        # Sort by distance ascending
        stmt = select(CourseChunk).order_by(CourseChunk.embedding.l2_distance(query_vector)).limit(limit)

        if course_id is not None:
            stmt = stmt.where(CourseChunk.course_id == course_id)
        if module_id is not None:
            stmt = stmt.where(CourseChunk.module_id == module_id)

        result = await self.db.execute(stmt)
        chunks = result.scalars().all()

        # Format results
        return [
            {
                "id": chunk.id,
                "score": 0.0, # Distance not easily available in scalar select unless requested separately, but irrelevant for simple retrieval
                "text": chunk.text,
                "course_id": chunk.course_id,
                "module_id": chunk.module_id,
                "module_type": chunk.module_type,
                "page_number": chunk.page_number,
                "heading": chunk.heading,
                "metadata": chunk.meta_data or {}
            }
            for chunk in chunks
        ]

    async def search_summaries(
        self,
        query: str,
        course_id: int,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Search specifically for community summaries using pgvector"""
        query_vector = self.embed_text(query)
        
        # Build query for summaries
        stmt = select(CourseChunk)\
            .where(CourseChunk.course_id == course_id)\
            .where(CourseChunk.module_type == "community_summary")\
            .order_by(CourseChunk.embedding.l2_distance(query_vector))\
            .limit(limit)
            
        result = await self.db.execute(stmt)
        chunks = result.scalars().all()
        
        return [
            {
                "id": chunk.id,
                "score": 0.0, # Distance calculation would require extra query complexity
                "text": chunk.text,
                "metadata": chunk.meta_data or {},
                "module_type": "community_summary"
            }
            for chunk in chunks
        ]

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Upsert documents into pgvector
        
        Args:
            documents: List of dicts with 'text', 'metadata', etc.
        """
        # Note: This is an insert/add operation. True upsert (on conflict update) would require specific handling.
        # For now we assume new chunks or we rely on ID handling if collisions occur.
        
        # Process in batches to optimize OpenAI calls
        batch_size = 100
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc["text"] for doc in batch]
            
            # Batch embedding generation
            embeddings = self.embed_texts(texts)
            
            for j, doc in enumerate(batch):
                doc_id = doc.get("id") or str(uuid.uuid4())
                meta = doc.get("metadata", {})
                
                chunk = CourseChunk(
                    id=doc_id,
                    text=doc["text"],
                    embedding=embeddings[j],
                    course_id=doc.get("course_id"),
                    module_id=doc.get("module_id"),
                    module_type=doc.get("module_type", "unknown"),
                    page_number=doc.get("page_number"),
                    heading=doc.get("heading"),
                    meta_data=meta,
                    timestamp_start=meta.get("timestamp_start"),
                    timestamp_end=meta.get("timestamp_end"),
                )
                await self.db.merge(chunk)
            
        await self.db.commit()
        return len(documents)
