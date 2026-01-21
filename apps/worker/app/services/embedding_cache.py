"""
Embedding Cache Service

Redis-based caching for OpenAI embeddings to avoid redundant API calls.
Uses a hash of the text as the key to store embedding vectors.

Features:
- Hash-based lookup (text -> embedding)
- TTL-based expiration (configurable, default 30 days)
- Cache hit rate metrics
- Batch operations support
"""
import hashlib
import json
import logging
from typing import List, Optional, Dict, Tuple
import redis
from ..config import config

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Redis-based cache for OpenAI embeddings.

    Reduces API costs and latency by caching embedding vectors.
    """

    def __init__(
        self,
        redis_url: str = None,
        ttl_days: int = 30,
        prefix: str = "emb:",
    ):
        """
        Initialize embedding cache.

        Args:
            redis_url: Redis connection URL
            ttl_days: Time-to-live for cached embeddings in days
            prefix: Key prefix for cache entries
        """
        self.redis_url = redis_url or config.REDIS_URL
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self.prefix = prefix
        self._client = None
        self._stats = {"hits": 0, "misses": 0}

    @property
    def client(self) -> redis.Redis:
        """Lazy initialization of Redis client."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # We store binary data
            )
        return self._client

    def _hash_text(self, text: str, model: str = "default") -> str:
        """
        Create a hash key for the text.

        Args:
            text: Text to hash
            model: Embedding model name (included in hash for model versioning)

        Returns:
            Hash string suitable for Redis key
        """
        # Normalize text (lowercase, strip whitespace)
        normalized = text.strip().lower()

        # Create hash including model name for version safety
        content = f"{model}:{normalized}"
        hash_value = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

        return f"{self.prefix}{hash_value}"

    def get(self, text: str, model: str = "default") -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Text to look up
            model: Embedding model name

        Returns:
            Embedding vector if cached, None otherwise
        """
        key = self._hash_text(text, model)

        try:
            data = self.client.get(key)
            if data:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit for embedding: {key[:20]}...")
                return json.loads(data)
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss for embedding: {key[:20]}...")
                return None
        except Exception as e:
            logger.warning(f"Error getting cached embedding: {e}")
            self._stats["misses"] += 1
            return None

    def set(
        self,
        text: str,
        embedding: List[float],
        model: str = "default",
        ttl: int = None,
    ) -> bool:
        """
        Cache an embedding for text.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model: Embedding model name
            ttl: Optional override for TTL in seconds

        Returns:
            True if cached successfully
        """
        key = self._hash_text(text, model)
        ttl = ttl or self.ttl_seconds

        try:
            data = json.dumps(embedding)
            self.client.setex(key, ttl, data)
            logger.debug(f"Cached embedding: {key[:20]}... (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
            return False

    def get_batch(
        self,
        texts: List[str],
        model: str = "default",
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts to look up
            model: Embedding model name

        Returns:
            Tuple of (cached_embeddings dict {index: embedding}, uncached_indices list)
        """
        if not texts:
            return {}, []

        keys = [self._hash_text(text, model) for text in texts]

        try:
            # Use pipeline for batch retrieval
            pipe = self.client.pipeline()
            for key in keys:
                pipe.get(key)
            results = pipe.execute()

            cached = {}
            uncached = []

            for i, data in enumerate(results):
                if data:
                    cached[i] = json.loads(data)
                    self._stats["hits"] += 1
                else:
                    uncached.append(i)
                    self._stats["misses"] += 1

            logger.info(f"Batch cache: {len(cached)} hits, {len(uncached)} misses")
            return cached, uncached

        except Exception as e:
            logger.warning(f"Error in batch cache lookup: {e}")
            return {}, list(range(len(texts)))

    def set_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model: str = "default",
        ttl: int = None,
    ) -> int:
        """
        Cache multiple embeddings.

        Args:
            texts: List of texts
            embeddings: Corresponding embedding vectors
            model: Embedding model name
            ttl: Optional TTL override

        Returns:
            Number of embeddings cached successfully
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")

        ttl = ttl or self.ttl_seconds
        cached_count = 0

        try:
            pipe = self.client.pipeline()
            for text, embedding in zip(texts, embeddings):
                key = self._hash_text(text, model)
                data = json.dumps(embedding)
                pipe.setex(key, ttl, data)

            results = pipe.execute()
            cached_count = sum(1 for r in results if r)

            logger.info(f"Batch cached {cached_count}/{len(texts)} embeddings")
            return cached_count

        except Exception as e:
            logger.warning(f"Error in batch cache set: {e}")
            return 0

    def delete(self, text: str, model: str = "default") -> bool:
        """
        Delete a cached embedding.

        Args:
            text: Text to invalidate
            model: Embedding model name

        Returns:
            True if deleted
        """
        key = self._hash_text(text, model)
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.warning(f"Error deleting cached embedding: {e}")
            return False

    def invalidate_all(self) -> int:
        """
        Delete all cached embeddings.

        Warning: This is a destructive operation.

        Returns:
            Number of entries deleted
        """
        try:
            pattern = f"{self.prefix}*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Error invalidating cache: {e}")
            return 0

    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dict with hit rate, total requests, etc.
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0

        # Get cache size
        try:
            pattern = f"{self.prefix}*"
            keys = list(self.client.scan_iter(match=pattern))
            cache_size = len(keys)
        except Exception:
            cache_size = -1

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "total_requests": total,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_percentage": f"{hit_rate * 100:.1f}%",
            "cache_size": cache_size,
            "ttl_days": self.ttl_seconds // (24 * 60 * 60),
        }

    def reset_stats(self):
        """Reset hit/miss statistics."""
        self._stats = {"hits": 0, "misses": 0}


# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache
