from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from app.core.config import settings
from app.core.database import get_db
from app.routers import courses, modules, assessment, reviews, chat, processing, adaptive, gamification, graph, social, curriculum, transformation, session



logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting NerdLearn API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")



    # Initialize Sentry if configured
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.ENVIRONMENT,
                traces_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 1.0,
            )
            logger.info("Sentry monitoring initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Sentry: {e}")

    yield

    # Shutdown
    logger.info("Shutting down NerdLearn API")



app = FastAPI(
    title="NerdLearn API",
    description="AI-Powered Adaptive Learning Platform API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(courses.router, prefix="/api/courses", tags=["courses"])
app.include_router(modules.router, prefix="/api/modules", tags=["modules"])
app.include_router(assessment.router, prefix="/api/assessment", tags=["assessment"])
app.include_router(reviews.router, prefix="/api/reviews", tags=["reviews"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(processing.router, prefix="/api/processing", tags=["processing"])
app.include_router(adaptive.router, prefix="/api/adaptive", tags=["adaptive"])
app.include_router(gamification.router, prefix="/api/gamification", tags=["gamification"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(social.router, prefix="/api/social", tags=["social"])
app.include_router(curriculum.router, prefix="/api/curriculum", tags=["curriculum"])
app.include_router(transformation.router, prefix="/api/transformation", tags=["transformation"])
app.include_router(session.router, prefix="/api/session", tags=["session"])


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "NerdLearn API",
            "version": "1.0.0",
            "docs": "/docs",
        }
    )


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint with database connectivity verification
    """
    health_status = {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
        "services": {}
    }

    # Check database connectivity
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["status"] = "degraded"
        health_status["services"]["database"] = "unhealthy"

    # Check Redis connectivity (for rate limiting)
    if settings.RATE_LIMIT_ENABLED:
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await redis_client.ping()
            await redis_client.close()
            health_status["services"]["redis"] = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            health_status["status"] = "degraded"
            health_status["services"]["redis"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
