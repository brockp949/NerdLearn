"""
Custom middleware for security, rate limiting, and monitoring
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from time import time
import redis.asyncio as redis
from loguru import logger
from app.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis"""

    def __init__(self, app, redis_url: str = None):
        super().__init__(app)
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client = None
        self.rate_limit = settings.RATE_LIMIT_PER_MINUTE
        self.enabled = settings.RATE_LIMIT_ENABLED

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host

        # Initialize Redis client if needed
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            except Exception as e:
                logger.error(f"Failed to connect to Redis for rate limiting: {e}")
                # If Redis is down, allow the request
                return await call_next(request)

        # Rate limit key
        key = f"rate_limit:{client_ip}"

        try:
            # Get current count
            current = await self.redis_client.get(key)

            if current is None:
                # First request in the window
                await self.redis_client.setex(key, 60, 1)
            else:
                current = int(current)
                if current >= self.rate_limit:
                    # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": "Too many requests. Please try again later."},
                    )
                else:
                    # Increment counter
                    await self.redis_client.incr(key)

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # On error, allow the request
            pass

        response = await call_next(request)
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""

    async def dispatch(self, request: Request, call_next):
        start_time = time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request failed: {request.method} {request.url.path} - Error: {str(e)}")
            raise

        # Calculate duration
        duration = time() - start_time

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration:.3f}s"
        )

        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        if settings.ENVIRONMENT == "production":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' https://api.openai.com;"
            )

        return response
