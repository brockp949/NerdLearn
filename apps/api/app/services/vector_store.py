"""
Vector Store Service for API
Manages similarity search using Qdrant
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import List, Dict, Any, Optional
import openai
from app.core.config import settings

class VectorStoreService:
    """Manages vector embeddings in Qdrant for the API"""

    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = settings.VECTOR_SIZE
        self.embedding_model = settings.EMBEDDING_MODEL
        openai.api_key = settings.OPENAI_API_KEY

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        response = openai.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def search(
        self,
        query: str,
        course_id: int = None,
        module_id: int = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in Qdrant"""
        # Generate query embedding
        query_vector = self.embed_text(query)

        # Build filter
        filter_conditions = {"must": []}
        if course_id is not None:
            filter_conditions["must"].append({"key": "course_id", "match": {"value": course_id}})
        if module_id is not None:
            filter_conditions["must"].append({"key": "module_id", "match": {"value": module_id}})

        if not filter_conditions["must"]:
            filter_conditions = None

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter_conditions,
            limit=limit,
        )

        # Format results
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text"),
                "course_id": hit.payload.get("course_id"),
                "module_id": hit.payload.get("module_id"),
                "module_type": hit.payload.get("module_type"),
                "page_number": hit.payload.get("page_number"),
                "heading": hit.payload.get("heading"),
                "metadata": {
                    "timestamp_start": hit.payload.get("timestamp_start"),
                    "timestamp_end": hit.payload.get("timestamp_end"),
                }
            }
            for hit in results
        ]
    async def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Upsert documents into Qdrant
        
        Args:
            documents: List of dicts with 'text', 'metadata', etc.
        """
        from qdrant_client.models import PointStruct
        import uuid

        points = []
        for doc in documents:
            text = doc["text"]
            # Generate ID if not provided
            doc_id = doc.get("id") or str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embed_text(text)
            
            # Prepare payload
            payload = {
                "text": text,
                **doc.get("metadata", {}),
                "course_id": doc.get("course_id"),
                "module_id": doc.get("module_id"),
                "module_type": doc.get("module_type", "unknown"),
            }
            
            points.append(PointStruct(id=doc_id, vector=embedding, payload=payload))
            
        # Upsert in batches (simplified for now)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)
