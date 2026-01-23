"""
Transformation API Router

Exposes content transformation services (Style Transfer, Diagrams, etc.)
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import uuid

from app.core.dspy_config import configure_dspy
from app.services.dspy_optimizer import StyleTransferModule

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["transformation"]
)


class StyleTransferRequest(BaseModel):
    """Request to rewrite content in a new style"""
    content: str = Field(..., description="Original educational content")
    target_persona: str = Field(..., description="Target style (ELI5, Academic, Socratic, etc.)")
    target_audience: str = Field(default="general learner", description="Description of the learner")
    preserve_facts: bool = Field(default=True, description="Whether to strictly verify fact retention")


class StyleTransferResponse(BaseModel):
    """Response from style transfer"""
    original_content: str
    transformed_content: str
    extracted_facts: List[str]
    persona: str
    fact_preservation_score: Optional[float] = None
    processing_time: float


@router.post("/style-transfer", response_model=StyleTransferResponse)
async def transfer_style(request: StyleTransferRequest):
    """
    Rewrite educational content into a specific persona using DSPy
    """
    start_time = datetime.utcnow()
    
    try:
        # Ensure DSPy is configured
        configure_dspy()
        
        # Initialize module
        pipeline = StyleTransferModule()
        
        # Execute pipeline
        logger.info(f"Starting style transfer to persona: {request.target_persona}")
        result = pipeline.forward(
            content=request.content,
            persona=request.target_persona,
            target_audience=request.target_audience,
            verify=request.preserve_facts
        )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "original_content": request.content,
            "transformed_content": result.transformed_content,
            "extracted_facts": result.facts,
            "persona": request.target_persona,
            "fact_preservation_score": result.score,
            "processing_time": duration
        }
        
    except Exception as e:
        logger.error(f"Style transfer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ================== Diagram Generation ==================

class DiagramRequest(BaseModel):
    """Request to generate a concept map"""
    content: str = Field(..., description="Educational content to visualize")
    topic: str = Field(default="", description="Optional topic context")


class DiagramResponse(BaseModel):
    """Response with graph structure"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    generated_at: datetime


@router.post("/diagram", response_model=DiagramResponse)
async def generate_diagram(request: DiagramRequest):
    """
    Generate a node-edge concept map from text
    """
    from app.services.diagram_service import DiagramService
    
    try:
        service = DiagramService()
        graph_data = await service.generate_concept_map(request.content, request.topic)
        
        return {
            "nodes": graph_data.get("nodes", []),
            "edges": graph_data.get("edges", []),
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Diagram generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ================== Podcast Generation ==================

class PodcastRequest(BaseModel):
    """Request to generate an audio podcast"""
    content: str = Field(..., description="Educational content source")
    topic: str = Field(default="Education", description="Topic context")


class PodcastResponse(BaseModel):
    """Response with script and audio link"""
    title: str
    script: List[Dict[str, str]]
    audio_url: str
    duration: float


@router.post("/podcast", response_model=PodcastResponse)
async def generate_podcast(request: PodcastRequest):
    """
    Generate an educational conversational podcast
    """
    from app.services.podcast_service import PodcastService
    
    try:
        service = PodcastService()
        result = await service.generate_podcast(request.content, request.topic)
        
        return result
        
    except Exception as e:
        logger.error(f"Podcast generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
