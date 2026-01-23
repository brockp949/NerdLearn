import logging
import time
from typing import Optional, Tuple
from fastapi import HTTPException, Request, Depends
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Redis-based Rate Limiter.
    Uses the fixed window algorithm for simplicity (or sliding window if keys utilize expiry).
    """
    def __init__(self, requests_per_minute: int = 60, redis_url: str = settings.REDIS_URL):
        self.limit = requests_per_minute
        self.redis_url = redis_url
        self._client: Optional[redis.Redis] = None

    async def get_client(self) -> redis.Redis:
        if not self._client:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def check_rate_limit(self, key: str) -> Tuple[bool, int]:
        """
        Check if rate limit is exceeded for the given key.
        Returns: (is_allowed, current_count)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, 0

        client = await self.get_client()
        window_start = int(time.time() // 60)
        redis_key = f"rate_limit:{key}:{window_start}"
        
        try:
            # Pipeline: increment and set expiry
            async with client.pipeline(transaction=True) as pipe:
                pipe.incr(redis_key)
                pipe.expire(redis_key, 90) # Expire after 90s to be safe
                result = await pipe.execute()
                
            count = result[0]
            if count > self.limit:
                logger.warning(f"Rate limit exceeded for {key}: {count}/{self.limit}")
                return False, count
            
            return True, count
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # Fail open if Redis is down
            return True, 0

    async def __call__(self, request: Request):
        """
        FastAPI Dependency for per-IP rate limiting.
        """
        if not settings.RATE_LIMIT_ENABLED:
            return

        client_ip = request.client.host if request.client else "unknown"
        is_allowed, count = await self.check_rate_limit(client_ip)
        
        if not is_allowed:
            raise HTTPException(
                status_code=429, 
                detail="Too Many Requests"
            )

# Default dependency
default_limiter = RateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)
