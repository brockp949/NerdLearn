"""
Vector Store Service
Manages embeddings and similarity search using pgvector
"""
import logging
import psycopg2
from psycopg2.extras import Json
from typing import List, Dict, Any, Optional
import openai
from ..config import config
from .embedding_cache import get_embedding_cache, EmbeddingCache
import uuid

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages vector embeddings with pgvector and caching support"""

    def __init__(self, use_cache: bool = True):
        """
        Initialize vector store service.
        """
        # Connection string
        self.dsn = config.DATABASE_URL
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

    def get_conn(self):
        return psycopg2.connect(self.dsn)

    def embed_text(self, text: str, use_cache: bool = None) -> List[float]:
        """Generate embedding using OpenAI"""
        should_cache = use_cache if use_cache is not None else self.use_cache

        if should_cache and self._cache:
            cached = self._cache.get(text, self.embedding_model)
            if cached:
                logger.debug("Using cached embedding")
                return cached

        response = openai.embeddings.create(input=[text], model=self.embedding_model)
        embedding = response.data[0].embedding

        if should_cache and self._cache:
            self._cache.set(text, embedding, self.embedding_model)

        return embedding

    def embed_batch(self, texts: List[str], use_cache: bool = None) -> List[List[float]]:
        if not texts:
            return []

        should_cache = use_cache if use_cache is not None else self.use_cache
        embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        if should_cache and self._cache:
            cached, uncached_indices = self._cache.get_batch(texts, self.embedding_model)
            for idx, emb in cached.items():
                embeddings[idx] = emb
            for idx in uncached_indices:
                texts_to_embed.append(texts[idx])
                indices_to_embed.append(idx)
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))

        if texts_to_embed:
            batch_size = 100
            new_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i : i + batch_size]
                response = openai.embeddings.create(
                    input=batch, model=self.embedding_model
                )
                batch_embeddings = [item.embedding for item in response.data]
                new_embeddings.extend(batch_embeddings)

            for i, idx in enumerate(indices_to_embed):
                embeddings[idx] = new_embeddings[i]

            if should_cache and self._cache:
                self._cache.set_batch(texts_to_embed, new_embeddings, self.embedding_model)

        return embeddings

    def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        course_id: int,
        module_id: int,
        module_type: str,
    ) -> List[str]:
        """Store document chunks with embeddings in pgvector"""
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_batch(texts)
        point_ids = []

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    point_id = str(uuid.uuid4())
                    point_ids.append(point_id)
                    
                    metadata = chunk.get("metadata", {})
                    
                    cur.execute(
                        """
                        INSERT INTO course_chunks 
                        (id, course_id, module_id, text, embedding, module_type, page_number, heading, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            point_id,
                            course_id,
                            module_id,
                            chunk["text"],
                            embedding, # psycopg2 + pgvector handles list -> vector
                            module_type,
                            chunk.get("page_number", metadata.get("page_number")),
                            chunk.get("heading"),
                            Json(metadata)
                        )
                    )
            conn.commit()

        return point_ids

    def search(
        self,
        query: str,
        course_id: int = None,
        module_id: int = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        query_vector = self.embed_text(query)
        
        where_clauses = []
        params = [query_vector] # First param is vector for L2 distance calculation
        
        if course_id is not None:
            where_clauses.append("course_id = %s")
            params.append(course_id)
        if module_id is not None:
            where_clauses.append("module_id = %s")
            params.append(module_id)

        where_sql = " AND ".join(where_clauses)
        if where_sql:
            where_sql = "WHERE " + where_sql
            
        sql = f"""
        SELECT id, text, course_id, module_id, module_type, page_number, heading, metadata
        FROM course_chunks
        {where_sql}
        ORDER BY embedding <-> %s
        LIMIT %s
        """
        # Note: params order: course_id, module_id, query_vector, limit
        # BUT wait: params list I built has query_vector FIRST.
        # ORDER BY uses query_vector.
        # If I use positional params, I must match the order in SQL.
        # But I put query_vector in ORDER BY clause which is at the END (or middle).
        
        # Correct param order:
        query_params = []
        if course_id is not None:
            query_params.append(course_id)
        if module_id is not None:
            query_params.append(module_id)
        query_params.append(query_vector) # for <->
        query_params.append(limit)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(query_params))
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    # id, text, course_id, module_id, module_type, page_number, heading, metadata
                    results.append({
                        "id": row[0],
                        "text": row[1],
                        "course_id": row[2],
                        "module_id": row[3],
                        "module_type": row[4],
                        "page_number": row[5],
                        "heading": row[6],
                        # metadata is row[7]
                    })
                return results

    def delete_module_chunks(self, module_id: int):
        with self.get_conn() as conn:
             with conn.cursor() as cur:
                 cur.execute("DELETE FROM course_chunks WHERE module_id = %s", (module_id,))
             conn.commit()

    def get_module_chunks(self, module_id: int) -> List[Dict[str, Any]]:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, text, metadata FROM course_chunks WHERE module_id = %s",
                    (module_id,)
                )
                rows = cur.fetchall()
                return [{"id": r[0], "text": r[1], "metadata": r[2]} for r in rows]
