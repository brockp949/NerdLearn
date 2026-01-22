"""
Multi-Modal Content Transformation Router

Research alignment:
- Podcastfy: Text-to-podcast synthesis
- React Flow: Interactive diagram generation
- Content Morphing: Seamless modality switching

This router enables the "Universal Translator for Complexity" vision -
transforming educational content across modalities while preserving
the learner's conceptual understanding state.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
import asyncio
import json

from app.multimodal import (
    PodcastGenerator,
    PodcastEpisode,
    SpeakerRole,
    get_podcast_generator,
    DiagramGenerator,
    DiagramType,
    DiagramData,
    get_diagram_generator,
    ContentMorpher,
    ContentModality,
    MorphedContent,
    ConceptualState,
    get_content_morpher
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Enums for API ==============

class PodcastStyle(str, Enum):
    educational = "educational"
    casual = "casual"
    debate = "debate"
    interview = "interview"


class DiagramTypeAPI(str, Enum):
    flowchart = "flowchart"
    mindmap = "mindmap"
    concept_map = "concept_map"
    sequence = "sequence"
    state = "state"


class ModalityType(str, Enum):
    text = "text"
    diagram = "diagram"
    podcast = "podcast"


# ============== Request/Response Schemas ==============

class PodcastGenerationRequest(BaseModel):
    """Request to generate a podcast from content"""
    content: str = Field(..., min_length=50, description="Educational content to transform")
    topic: str = Field(..., description="Topic title for the podcast")
    duration_minutes: int = Field(default=10, ge=3, le=30, description="Target duration in minutes")
    style: PodcastStyle = Field(default=PodcastStyle.educational)
    include_expert: bool = Field(default=True, description="Include expert guest speaker")
    use_debate_format: bool = Field(default=False, description="Use skeptic for debate format")


class PodcastGenerationResponse(BaseModel):
    """Response with generated podcast"""
    episode_id: str
    title: str
    script_segments: List[Dict[str, Any]]
    total_duration_seconds: int
    audio_url: Optional[str] = None
    concepts_covered: List[str]


class DiagramGenerationRequest(BaseModel):
    """Request to generate a diagram from content"""
    content: str = Field(..., min_length=20, description="Content to visualize")
    diagram_type: DiagramTypeAPI = Field(default=DiagramTypeAPI.concept_map)
    title: Optional[str] = Field(default=None, description="Diagram title")
    focus_concepts: Optional[List[str]] = Field(default=None, description="Concepts to highlight")


class DiagramGenerationResponse(BaseModel):
    """Response with generated diagram in React Flow format"""
    diagram_id: str
    diagram_type: str
    title: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    mermaid_source: str
    metadata: Dict[str, Any]


class ConceptMapRequest(BaseModel):
    """Request to generate concept map from explicit concepts"""
    concepts: List[Dict[str, Any]] = Field(..., description="List of concepts with id, name, type")
    relationships: List[Dict[str, Any]] = Field(..., description="List of relationships with source, target, label")
    title: str = Field(default="Concept Map")


class ContentMorphRequest(BaseModel):
    """Request to transform content between modalities"""
    content: str = Field(..., description="Source content to transform")
    source_modality: ModalityType = Field(..., description="Current modality of content")
    target_modality: ModalityType = Field(..., description="Desired modality")
    user_id: Optional[str] = Field(default=None, description="User ID for state persistence")
    content_id: Optional[str] = Field(default=None, description="Content ID for state persistence")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Transformation options")


class ContentMorphResponse(BaseModel):
    """Response with transformed content"""
    original_modality: str
    target_modality: str
    content: Any
    concepts_extracted: List[str]
    transformation_notes: str
    state_updated: bool


class ModalityRecommendationRequest(BaseModel):
    """Request for modality recommendation"""
    user_id: str = Field(..., description="User ID")
    content_id: str = Field(..., description="Content ID")
    current_concept: Optional[str] = Field(default=None, description="Current concept being studied")
    device: Optional[str] = Field(default=None, description="Device type (mobile, desktop)")
    time_available_minutes: Optional[int] = Field(default=None, description="Available study time")


class ModalityRecommendationResponse(BaseModel):
    """Response with modality recommendation"""
    recommended_modality: str
    reason: str
    alternatives: List[str]
    weak_concepts: List[str]
    current_state: str


class LearningSummaryResponse(BaseModel):
    """Response with learning progress summary"""
    user_id: str
    content_id: str
    total_concepts: int
    progress_percent: float
    mastery_distribution: Dict[str, int]
    modality_usage: Dict[str, int]
    session_duration_seconds: float
    modality_switches: int
    weak_concepts: List[str]
    current_modality: str


# ============== Podcast Endpoints ==============

@router.post("/podcast/generate", response_model=PodcastGenerationResponse)
async def generate_podcast(
    request: PodcastGenerationRequest,
    podcast_gen: PodcastGenerator = Depends(get_podcast_generator)
):
    """
    Generate a podcast episode from educational content

    Transforms text content into an engaging multi-speaker podcast
    with natural dialogue, explanations, and (optionally) synthesized audio.
    """
    try:
        logger.info(f"Generating podcast for topic: {request.topic}")

        episode = await podcast_gen.generate(
            content=request.content,
            topic=request.topic,
            duration_minutes=request.duration_minutes,
            style=request.style.value,
            include_expert=request.include_expert,
            use_debate_format=request.use_debate_format
        )

        return PodcastGenerationResponse(
            episode_id=episode.id,
            title=episode.title,
            script_segments=[
                {
                    "speaker": seg.speaker.value,
                    "text": seg.text,
                    "duration_seconds": seg.duration_seconds,
                    "emotion": seg.emotion
                }
                for seg in episode.script.segments
            ],
            total_duration_seconds=episode.duration_seconds,
            audio_url=episode.audio_url,
            concepts_covered=episode.metadata.get("concepts", [])
        )

    except Exception as e:
        logger.error(f"Error generating podcast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/podcast/generate/async")
async def generate_podcast_async(
    request: PodcastGenerationRequest,
    background_tasks: BackgroundTasks,
    podcast_gen: PodcastGenerator = Depends(get_podcast_generator)
):
    """
    Start async podcast generation (for longer content with audio synthesis)

    Returns a job ID to track progress.
    """
    import uuid
    job_id = str(uuid.uuid4())

    # In production, store job status in Redis/DB
    background_tasks.add_task(
        _generate_podcast_background,
        job_id,
        request,
        podcast_gen
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Podcast generation started",
        "estimated_time_seconds": request.duration_minutes * 10
    }


async def _generate_podcast_background(
    job_id: str,
    request: PodcastGenerationRequest,
    podcast_gen: PodcastGenerator
):
    """Background task for podcast generation"""
    try:
        episode = await podcast_gen.generate(
            content=request.content,
            topic=request.topic,
            duration_minutes=request.duration_minutes,
            style=request.style.value
        )
        # Store result (in production, use Redis/DB)
        logger.info(f"Podcast job {job_id} completed")
    except Exception as e:
        logger.error(f"Podcast job {job_id} failed: {e}")


# ============== Diagram Endpoints ==============

@router.post("/diagram/generate", response_model=DiagramGenerationResponse)
async def generate_diagram(
    request: DiagramGenerationRequest,
    diagram_gen: DiagramGenerator = Depends(get_diagram_generator)
):
    """
    Generate an interactive diagram from content

    Creates a React Flow compatible diagram with nodes and edges
    that can be rendered in the web UI.
    """
    try:
        logger.info(f"Generating {request.diagram_type.value} diagram")

        # Map API enum to internal enum
        diagram_type_map = {
            DiagramTypeAPI.flowchart: DiagramType.FLOWCHART,
            DiagramTypeAPI.mindmap: DiagramType.MINDMAP,
            DiagramTypeAPI.concept_map: DiagramType.CONCEPT_MAP,
            DiagramTypeAPI.sequence: DiagramType.SEQUENCE,
            DiagramTypeAPI.state: DiagramType.STATE,
        }

        diagram = await diagram_gen.generate(
            content=request.content,
            diagram_type=diagram_type_map[request.diagram_type],
            title=request.title,
            focus_concepts=request.focus_concepts
        )

        react_flow_data = diagram.to_react_flow()

        return DiagramGenerationResponse(
            diagram_id=react_flow_data["id"],
            diagram_type=react_flow_data["type"],
            title=react_flow_data["title"],
            nodes=react_flow_data["nodes"],
            edges=react_flow_data["edges"],
            mermaid_source=react_flow_data["mermaidSource"],
            metadata=react_flow_data["metadata"]
        )

    except Exception as e:
        logger.error(f"Error generating diagram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diagram/from-concepts", response_model=DiagramGenerationResponse)
async def generate_diagram_from_concepts(
    request: ConceptMapRequest,
    diagram_gen: DiagramGenerator = Depends(get_diagram_generator)
):
    """
    Generate a concept map directly from concept and relationship data

    Useful when you already have structured concept data from
    the knowledge graph.
    """
    try:
        diagram = await diagram_gen.generate_from_concepts(
            concepts=request.concepts,
            relationships=request.relationships,
            title=request.title
        )

        react_flow_data = diagram.to_react_flow()

        return DiagramGenerationResponse(
            diagram_id=react_flow_data["id"],
            diagram_type=react_flow_data["type"],
            title=react_flow_data["title"],
            nodes=react_flow_data["nodes"],
            edges=react_flow_data["edges"],
            mermaid_source=react_flow_data["mermaidSource"],
            metadata=react_flow_data["metadata"]
        )

    except Exception as e:
        logger.error(f"Error generating concept map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagram/mermaid-preview")
async def preview_mermaid(
    content: str,
    diagram_type: DiagramTypeAPI = DiagramTypeAPI.flowchart,
    diagram_gen: DiagramGenerator = Depends(get_diagram_generator)
):
    """
    Quick preview - returns just the Mermaid syntax

    Useful for previewing in Mermaid Live Editor or debugging.
    """
    try:
        diagram_type_map = {
            DiagramTypeAPI.flowchart: DiagramType.FLOWCHART,
            DiagramTypeAPI.mindmap: DiagramType.MINDMAP,
            DiagramTypeAPI.concept_map: DiagramType.CONCEPT_MAP,
            DiagramTypeAPI.sequence: DiagramType.SEQUENCE,
            DiagramTypeAPI.state: DiagramType.STATE,
        }

        mermaid = await diagram_gen.generate_mermaid(
            content=content,
            diagram_type=diagram_type_map[diagram_type]
        )

        return {
            "mermaid": mermaid,
            "diagram_type": diagram_type.value,
            "live_editor_url": f"https://mermaid.live/edit"
        }

    except Exception as e:
        logger.error(f"Error generating Mermaid preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Content Morphing Endpoints ==============

@router.post("/morph", response_model=ContentMorphResponse)
async def morph_content(
    request: ContentMorphRequest,
    morpher: ContentMorpher = Depends(get_content_morpher)
):
    """
    Transform content between modalities (text, diagram, podcast)

    Maintains conceptual state across transformations, enabling
    learners to switch formats without losing progress.
    """
    try:
        logger.info(f"Morphing: {request.source_modality.value} -> {request.target_modality.value}")

        # Map API enums to internal enums
        modality_map = {
            ModalityType.text: ContentModality.TEXT,
            ModalityType.diagram: ContentModality.DIAGRAM,
            ModalityType.podcast: ContentModality.PODCAST,
        }

        result = await morpher.morph(
            content=request.content,
            source_modality=modality_map[request.source_modality],
            target_modality=modality_map[request.target_modality],
            user_id=request.user_id,
            content_id=request.content_id,
            options=request.options
        )

        # Serialize content based on type
        serialized_content = result.content
        if hasattr(result.content, 'to_dict'):
            serialized_content = result.content.to_dict()
        elif hasattr(result.content, 'to_react_flow'):
            serialized_content = result.content.to_react_flow()

        return ContentMorphResponse(
            original_modality=result.original_modality.value,
            target_modality=result.target_modality.value,
            content=serialized_content,
            concepts_extracted=result.concepts_extracted,
            transformation_notes=result.transformation_notes,
            state_updated=result.conceptual_state is not None
        )

    except Exception as e:
        logger.error(f"Error morphing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-modality", response_model=ModalityRecommendationResponse)
async def recommend_modality(
    request: ModalityRecommendationRequest,
    morpher: ContentMorpher = Depends(get_content_morpher)
):
    """
    Get AI-powered modality recommendation based on learning state

    Considers:
    - Concept mastery levels
    - Modality preferences from history
    - Current context (device, available time)
    """
    try:
        context = {}
        if request.device:
            context["device"] = request.device
        if request.time_available_minutes:
            context["time_available_minutes"] = request.time_available_minutes

        recommendation = await morpher.recommend_modality(
            user_id=request.user_id,
            content_id=request.content_id,
            current_concept=request.current_concept,
            context=context
        )

        return ModalityRecommendationResponse(**recommendation)

    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning-summary/{user_id}/{content_id}", response_model=LearningSummaryResponse)
async def get_learning_summary(
    user_id: str,
    content_id: str,
    morpher: ContentMorpher = Depends(get_content_morpher)
):
    """
    Get learning progress summary across all modalities

    Shows concept mastery, modality usage, and areas needing attention.
    """
    try:
        summary = await morpher.get_learning_summary(user_id, content_id)
        return LearningSummaryResponse(**summary)

    except Exception as e:
        logger.error(f"Error getting learning summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{user_id}/{content_id}")
async def get_conceptual_state(
    user_id: str,
    content_id: str,
    morpher: ContentMorpher = Depends(get_content_morpher)
):
    """
    Get raw conceptual state for debugging/inspection

    Returns the full state object tracking concept mastery
    and modality history.
    """
    try:
        state = morpher.get_or_create_state(user_id, content_id)
        return state.to_dict()

    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/state/{user_id}/{content_id}")
async def reset_conceptual_state(
    user_id: str,
    content_id: str,
    morpher: ContentMorpher = Depends(get_content_morpher)
):
    """
    Reset conceptual state (start fresh)

    Useful for testing or when learner wants to restart.
    """
    try:
        key = morpher._get_state_key(user_id, content_id)
        if key in morpher._state_cache:
            del morpher._state_cache[key]
        return {"message": "State reset successfully", "user_id": user_id, "content_id": content_id}

    except Exception as e:
        logger.error(f"Error resetting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Utility Endpoints ==============

@router.get("/modalities")
async def list_supported_modalities():
    """
    List all supported content modalities

    Returns capabilities and recommended use cases for each.
    """
    return {
        "modalities": [
            {
                "id": "text",
                "name": "Text",
                "description": "Traditional written content with formatting",
                "best_for": ["Quick review", "Reference", "Note-taking"],
                "supported": True
            },
            {
                "id": "diagram",
                "name": "Interactive Diagram",
                "description": "Visual concept maps and flowcharts",
                "best_for": ["Understanding relationships", "System overview", "Visual learners"],
                "supported": True
            },
            {
                "id": "podcast",
                "name": "Podcast",
                "description": "Audio explanations with multiple speakers",
                "best_for": ["Deep understanding", "Passive learning", "Commute/exercise"],
                "supported": True
            },
            {
                "id": "video",
                "name": "Video",
                "description": "Animated explanations",
                "best_for": ["Complex processes", "Step-by-step tutorials"],
                "supported": False,
                "coming_soon": True
            },
            {
                "id": "interactive",
                "name": "Interactive Exercises",
                "description": "Hands-on practice and simulations",
                "best_for": ["Skill building", "Active recall"],
                "supported": False,
                "coming_soon": True
            }
        ]
    }


@router.get("/diagram-types")
async def list_diagram_types():
    """
    List all supported diagram types with examples
    """
    return {
        "diagram_types": [
            {
                "id": "flowchart",
                "name": "Flowchart",
                "description": "Process flows, algorithms, decision trees",
                "example": "flowchart TD\n    A[Start] --> B{Decision}\n    B -->|Yes| C[Action]"
            },
            {
                "id": "mindmap",
                "name": "Mind Map",
                "description": "Hierarchical topic exploration",
                "example": "mindmap\n  root((Topic))\n    Branch1\n      Leaf1"
            },
            {
                "id": "concept_map",
                "name": "Concept Map",
                "description": "Relationships between ideas",
                "example": "flowchart LR\n    A[Concept] -->|relates to| B[Concept]"
            },
            {
                "id": "sequence",
                "name": "Sequence Diagram",
                "description": "Interactions and message flows",
                "example": "sequenceDiagram\n    A->>B: Request"
            },
            {
                "id": "state",
                "name": "State Diagram",
                "description": "State machines and transitions",
                "example": "stateDiagram-v2\n    [*] --> State1"
            }
        ]
    }
