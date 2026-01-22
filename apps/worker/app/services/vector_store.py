"""
Vector Store Service
Manages embeddings and similarity search using Qdrant

Enhanced with embedding cache to reduce OpenAI API costs.
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import openai
from ..config import config
from .embedding_cache import get_embedding_cache, EmbeddingCache
import uuid

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages vector embeddings in Qdrant with caching support"""

    def __init__(self, use_cache: bool = True):
        """
        Initialize vector store service.

        Args:
            use_cache: Whether to use embedding cache (default True)
        """
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.collection_name = config.COLLECTION_NAME
        self.vector_size = config.VECTOR_SIZE
        self.embedding_model = config.EMBEDDING_MODEL
        openai.api_key = config.OPENAI_API_KEY

        # Initialize cache
        self.use_cache = use_cache
        self._cache: Optional[EmbeddingCache] = None
        if use_cache:
            try:
                self._cache = get_embedding_cache()
                logger.info("Embedding cache enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding cache: {e}")
                self.use_cache = False

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            print(f"Created collection: {self.collection_name}")

    def embed_text(self, text: str, use_cache: bool = None) -> List[float]:
        """
        Generate embedding for text using OpenAI with optional caching.

        Args:
            text: Text to embed
            use_cache: Override cache setting for this call

        Returns:
            Embedding vector
        """
        should_cache = use_cache if use_cache is not None else self.use_cache

        # Try cache first
        if should_cache and self._cache:
            cached = self._cache.get(text, self.embedding_model)
            if cached:
                logger.debug("Using cached embedding")
                return cached

        # Generate embedding via API
        response = openai.embeddings.create(input=[text], model=self.embedding_model)
        embedding = response.data[0].embedding

        # Cache the result
        if should_cache and self._cache:
            self._cache.set(text, embedding, self.embedding_model)

        return embedding

    def embed_batch(self, texts: List[str], use_cache: bool = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching support.

        Uses cache for already-embedded texts and only calls API for new texts.

        Args:
            texts: List of texts to embed
            use_cache: Override cache setting for this call

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        should_cache = use_cache if use_cache is not None else self.use_cache

        # Initialize result array
        embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        # Check cache for existing embeddings
        if should_cache and self._cache:
            cached, uncached_indices = self._cache.get_batch(texts, self.embedding_model)

            # Fill in cached embeddings
            for idx, emb in cached.items():
                embeddings[idx] = emb

            # Collect texts that need embedding
            for idx in uncached_indices:
                texts_to_embed.append(texts[idx])
                indices_to_embed.append(idx)

            cache_hits = len(cached)
            logger.info(f"Embedding batch: {cache_hits} cached, {len(texts_to_embed)} to generate")
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))

        # Generate embeddings for uncached texts
        if texts_to_embed:
            # OpenAI has a limit on batch size
            batch_size = 100
            new_embeddings = []

            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i : i + batch_size]
                response = openai.embeddings.create(
                    input=batch, model=self.embedding_model
                )
                batch_embeddings = [item.embedding for item in response.data]
                new_embeddings.extend(batch_embeddings)

            # Fill in new embeddings at correct indices
            for i, idx in enumerate(indices_to_embed):
                embeddings[idx] = new_embeddings[i]

            # Cache new embeddings
            if should_cache and self._cache:
                self._cache.set_batch(texts_to_embed, new_embeddings, self.embedding_model)

        return embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get embedding cache statistics.

        Returns:
            Dict with cache hit rate, size, etc.
        """
        if self._cache:
            return self._cache.get_stats()
        return {"cache_enabled": False}

    def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        course_id: int,
        module_id: int,
        module_type: str,
    ) -> List[str]:
        """
        Store document chunks with embeddings in Qdrant

        Args:
            chunks: List of text chunks with metadata
            course_id: Course ID
            module_id: Module ID
            module_type: Type of module (pdf, video)

        Returns:
            List of point IDs
        """
        if not chunks:
            return []

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embed_batch(texts)

        # Create points for Qdrant
        points = []
        point_ids = []

        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            payload = {
                "course_id": course_id,
                "module_id": module_id,
                "module_type": module_type,
                "chunk_id": chunk.get("chunk_id", 0),
                "text": chunk["text"],
                "token_count": chunk.get("token_count", 0),
                "heading": chunk.get("heading"),
                "page_number": chunk.get("metadata", {}).get("page_number"),
                **chunk.get("metadata", {}),
            }

            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        # Upload to Qdrant
        print(f"Uploading {len(points)} points to Qdrant...")
        self.client.upsert(collection_name=self.collection_name, points=points)

        return point_ids

    def search(
        self,
        query: str,
        course_id: int = None,
        module_id: int = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks

        Args:
            query: Search query
            course_id: Optional course filter
            module_id: Optional module filter
            limit: Number of results

        Returns:
            List of matching chunks with scores
        """
        # Generate query embedding
        query_vector = self.embed_text(query)

        # Build filter
        filter_conditions = None
        if course_id is not None:
            filter_conditions = {"must": [{"key": "course_id", "match": {"value": course_id}}]}
            if module_id is not None:
                filter_conditions["must"].append(
                    {"key": "module_id", "match": {"value": module_id}}
                )

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
            }
            for hit in results
        ]

    def delete_module_chunks(self, module_id: int):
        """Delete all chunks for a module"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"filter": {"must": [{"key": "module_id", "match": {"value": module_id}}]}},
        )
