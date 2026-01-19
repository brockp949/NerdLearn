"""
Knowledge Graph Router
API endpoints for knowledge graph operations

Research alignment:
- Knowledge Graph Construction for Educational Systems
- Prerequisite extraction and learning path generation
- Graph-based adaptive content sequencing
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.services.graph_service import get_graph_service, AsyncGraphService

router = APIRouter()


# ============== Request/Response Models ==============

class PrerequisiteCreate(BaseModel):
    """Request model for creating prerequisite relationships"""
    prerequisite_name: str = Field(..., description="Name of the prerequisite concept")
    concept_name: str = Field(..., description="Name of the dependent concept")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Confidence score")
    prereq_type: str = Field(default="explicit", description="Type: explicit, sequential, inferred")


class ConceptCreate(BaseModel):
    """Request model for creating concepts"""
    name: str = Field(..., description="Concept name")
    module_id: int = Field(..., description="Module ID to associate with")
    difficulty: float = Field(default=5.0, ge=1, le=10, description="Difficulty 1-10")
    importance: float = Field(default=0.5, ge=0, le=1, description="Importance 0-1")
    description: Optional[str] = Field(default=None, description="Concept description")


class LearningPathRequest(BaseModel):
    """Request model for learning path generation"""
    target_concepts: List[str] = Field(..., description="Concepts to learn")
    mastered_concepts: Optional[List[str]] = Field(default=None, description="Already mastered")


class GraphNode(BaseModel):
    """Graph node representation"""
    id: str
    label: str
    module: Optional[str] = None
    module_id: Optional[int] = None
    module_order: int = 0
    difficulty: float = 5.0
    importance: float = 0.5
    type: str = "concept"


class GraphEdge(BaseModel):
    """Graph edge representation"""
    source: str
    target: str
    type: str = "prerequisite"
    confidence: float = 0.5


class GraphResponse(BaseModel):
    """Full graph response"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    meta: dict


class ConceptExtractionResponse(BaseModel):
    """Response model for concept extraction"""
    concepts: List[str]
    count: int
    method: str


class LegacyGraphResponse(BaseModel):
    """Legacy response for force-graph compatibility"""
    nodes: List[dict]
    links: List[dict]


# ============== Endpoints ==============

@router.get("/courses/{course_id}", response_model=GraphResponse)
async def get_course_graph(
    course_id: int,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Get the complete knowledge graph for a course.

    Returns all concepts and their prerequisite relationships
    for visualization and navigation.
    """
    try:
        graph = await graph_service.get_course_graph(course_id)

        if not graph["nodes"]:
            # Return empty graph structure instead of error
            return GraphResponse(
                nodes=[],
                edges=[],
                meta={"course_id": course_id, "total_concepts": 0, "total_relationships": 0}
            )

        return GraphResponse(
            nodes=[GraphNode(**n) for n in graph["nodes"]],
            edges=[GraphEdge(**e) for e in graph["edges"]],
            meta=graph["meta"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph: {str(e)}")


@router.get("/courses/{course_id}/stats")
async def get_graph_stats(
    course_id: int,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Get statistics about the knowledge graph.

    Returns counts of modules, concepts, prerequisites, and average difficulty.
    """
    try:
        stats = await graph_service.get_graph_stats(course_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.get("/courses/{course_id}/concepts/{concept_name}")
async def get_concept_details(
    course_id: int,
    concept_name: str,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Get detailed information about a specific concept.

    Includes prerequisites, dependent concepts, and metadata.
    """
    try:
        details = await graph_service.get_concept_details(course_id, concept_name)

        if not details:
            raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")

        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch concept: {str(e)}")


@router.get("/courses/{course_id}/concepts/{concept_name}/prerequisites")
async def get_concept_prerequisites(
    course_id: int,
    concept_name: str,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Get all prerequisites for a concept.

    Returns list of concepts that must be learned before this one.
    """
    try:
        prerequisites = await graph_service.get_prerequisites(course_id, concept_name)
        return {"concept": concept_name, "prerequisites": prerequisites}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prerequisites: {str(e)}")


@router.get("/courses/{course_id}/concepts/{concept_name}/dependents")
async def get_concept_dependents(
    course_id: int,
    concept_name: str,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Get all concepts that depend on this concept.

    Returns list of concepts that have this as a prerequisite.
    """
    try:
        dependents = await graph_service.get_dependents(course_id, concept_name)
        return {"concept": concept_name, "dependents": dependents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dependents: {str(e)}")


@router.post("/courses/{course_id}/learning-path")
async def generate_learning_path(
    course_id: int,
    request: LearningPathRequest,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Generate an optimal learning path to reach target concepts.

    Uses prerequisite graph to determine the order of concepts
    to learn, excluding already mastered concepts.

    Research basis: Graph-based adaptive sequencing ensures
    prerequisites are satisfied before advancing.
    """
    try:
        learning_path = await graph_service.get_learning_path(
            course_id=course_id,
            target_concepts=request.target_concepts,
            user_mastered=request.mastered_concepts
        )

        return {
            "target_concepts": request.target_concepts,
            "mastered_concepts": request.mastered_concepts or [],
            "learning_path": learning_path,
            "total_concepts": len(learning_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate path: {str(e)}")


@router.get("/courses/{course_id}/entry-points")
async def get_entry_points(
    course_id: int,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Get concepts with no prerequisites (entry points).

    These are good starting points for learning.
    """
    try:
        entry_points = await graph_service.find_concepts_without_prerequisites(course_id)
        return {"course_id": course_id, "entry_points": entry_points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch entry points: {str(e)}")


@router.get("/courses/{course_id}/terminal-concepts")
async def get_terminal_concepts(
    course_id: int,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Get concepts that are not prerequisites for anything (endpoints).

    These represent advanced or final topics in the course.
    """
    try:
        terminal = await graph_service.find_terminal_concepts(course_id)
        return {"course_id": course_id, "terminal_concepts": terminal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch terminal concepts: {str(e)}")


# ============== Mutation Endpoints ==============

@router.post("/courses/{course_id}/concepts")
async def create_concept(
    course_id: int,
    concept: ConceptCreate,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Create a new concept in the knowledge graph.

    Links the concept to the specified module.
    """
    try:
        success = await graph_service.create_concept_node(
            course_id=course_id,
            module_id=concept.module_id,
            name=concept.name,
            difficulty=concept.difficulty,
            importance=concept.importance,
            description=concept.description
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to create concept")

        return {"status": "created", "concept": concept.name, "course_id": course_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create concept: {str(e)}")


@router.post("/courses/{course_id}/prerequisites")
async def add_prerequisite(
    course_id: int,
    prereq: PrerequisiteCreate,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Add a prerequisite relationship between concepts.

    Creates a directed edge from prerequisite to dependent concept.
    """
    try:
        success = await graph_service.add_prerequisite(
            course_id=course_id,
            prerequisite_name=prereq.prerequisite_name,
            concept_name=prereq.concept_name,
            confidence=prereq.confidence,
            prereq_type=prereq.prereq_type
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to create prerequisite. Check that both concepts exist."
            )

        return {
            "status": "created",
            "prerequisite": prereq.prerequisite_name,
            "concept": prereq.concept_name,
            "confidence": prereq.confidence
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add prerequisite: {str(e)}")


@router.delete("/courses/{course_id}/prerequisites")
async def remove_prerequisite(
    course_id: int,
    prerequisite_name: str = Query(..., description="Prerequisite concept name"),
    concept_name: str = Query(..., description="Dependent concept name"),
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Remove a prerequisite relationship between concepts.
    """
    try:
        success = await graph_service.remove_prerequisite(
            course_id=course_id,
            prerequisite_name=prerequisite_name,
            concept_name=concept_name
        )

        if not success:
            raise HTTPException(status_code=404, detail="Prerequisite relationship not found")

        return {
            "status": "deleted",
            "prerequisite": prerequisite_name,
            "concept": concept_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove prerequisite: {str(e)}")


@router.post("/courses/{course_id}/detect-prerequisites")
async def detect_prerequisites(
    course_id: int,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Auto-detect prerequisite relationships based on module ordering.

    Uses sequential heuristic: concepts in earlier modules are
    prerequisites for concepts in later modules (confidence: 0.3).

    Research basis: Module sequencing often reflects pedagogical
    prerequisite ordering.
    """
    try:
        count = await graph_service.detect_prerequisites_sequential(course_id)
        return {
            "status": "completed",
            "course_id": course_id,
            "relationships_created": count,
            "detection_method": "sequential",
            "confidence": 0.3
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect prerequisites: {str(e)}")


# ============== Utility Endpoints ==============

@router.post("/extract-concepts", response_model=ConceptExtractionResponse)
async def extract_concepts_from_text(
    text: str = Query(..., min_length=10, description="Text to extract concepts from"),
    min_freq: int = Query(default=1, ge=1, description="Minimum frequency threshold"),
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Extract concepts from text using NLP heuristics.

    Uses pattern matching and technical term detection.
    For production, integrate with BERT-NER (92.8% F1 per research).
    """
    try:
        concepts = graph_service.extract_concepts(text, min_freq)
        return ConceptExtractionResponse(
            concepts=concepts,
            count=len(concepts),
            method="pattern_matching"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract concepts: {str(e)}")


# Legacy endpoint for backwards compatibility
@router.get("/", response_model=LegacyGraphResponse)
async def get_graph_legacy(
    course_id: Optional[int] = Query(default=1, description="Course ID (defaults to 1)"),
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Legacy endpoint for frontend compatibility.

    Returns graph with 'links' key (instead of 'edges') for react-force-graph compatibility.
    Defaults to course_id=1 for backward compatibility.

    Use /courses/{course_id} for new implementations.
    """
    try:
        graph = await graph_service.get_course_graph(course_id)

        # Transform edges to links for frontend compatibility
        links = [
            {
                "source": edge["source"],
                "target": edge["target"],
                "type": edge.get("type", "prerequisite"),
                "confidence": edge.get("confidence", 0.5),
            }
            for edge in graph.get("edges", [])
        ]

        return LegacyGraphResponse(
            nodes=graph.get("nodes", []),
            links=links
        )
    except Exception as e:
        # Log the error here in a real app (e.g. logger.error(f"Legacy graph error: {e}"))
        # Return empty graph on error to prevent frontend crash
        return LegacyGraphResponse(nodes=[], links=[])
