from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.routers import courses, modules, assessment, reviews, chat, processing, adaptive, gamification, graph
from app.services.graph_service import graph_service
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events"""
    # Startup
    try:
        await graph_service.connect()
        logger.info("Neo4j connection established")
    except Exception as e:
        logger.warning(f"Neo4j connection failed (service may not be running): {e}")

    yield

    # Shutdown
    try:
        await graph_service.close()
        logger.info("Neo4j connection closed")
    except Exception as e:
        logger.warning(f"Error closing Neo4j connection: {e}")


app = FastAPI(
    title="NerdLearn API",
    description="AI-Powered Adaptive Learning Platform API",
    version="0.1.0",
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


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "NerdLearn API",
            "version": "0.1.0",
            "docs": "/docs",
        }
    )


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})
