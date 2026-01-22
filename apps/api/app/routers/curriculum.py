"""
Curriculum Generation Router - Agentic Curriculum Generation API

Research alignment:
- HiPlan: Hierarchical multi-agent planning
- Architect → Refiner → Verifier workflow
- LangGraph cyclic state management
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
from enum import Enum

from app.agents import AgentGraph, CurriculumConstraints
from app.services.graph_service import get_graph_service, AsyncGraphService

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Request/Response Schemas ==============

class DifficultyLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class LearningStyle(str, Enum):
    visual = "visual"
    text = "text"
    interactive = "interactive"
    balanced = "balanced"


class CurriculumGenerationRequest(BaseModel):
    """Request to generate a curriculum"""
    topic: str = Field(..., description="Main topic for the curriculum (e.g., 'Machine Learning')")
    course_id: int = Field(..., description="Course ID to associate with")
    duration_weeks: int = Field(default=4, ge=1, le=16, description="Course duration in weeks")
    difficulty_level: DifficultyLevel = Field(default=DifficultyLevel.intermediate)
    target_audience: str = Field(default="general", description="Target learner persona")
    prerequisites: List[str] = Field(default_factory=list, description="Required prior knowledge")
    learning_style: LearningStyle = Field(default=LearningStyle.balanced)
    max_modules: Optional[int] = Field(default=None, ge=1, le=20, description="Maximum number of modules")

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Introduction to Machine Learning",
                "course_id": 1,
                "duration_weeks": 4,
                "difficulty_level": "intermediate",
                "target_audience": "software developers",
                "prerequisites": ["Python programming", "Basic statistics"],
                "learning_style": "balanced"
            }
        }


class CurriculumGenerationResponse(BaseModel):
    """Response from curriculum generation"""
    success: bool
    job_id: Optional[str] = None
    syllabus: Optional[Dict[str, Any]] = None
    quality_score: Optional[int] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    generation_time_seconds: Optional[float] = None


class CurriculumJobStatus(BaseModel):
    """Status of an async curriculum generation job"""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: int  # 0-100
    current_agent: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# In-memory job storage (replace with Redis in production)
_curriculum_jobs: Dict[str, Dict[str, Any]] = {}


# ============== API Endpoints ==============

@router.post("/generate", response_model=CurriculumGenerationResponse)
async def generate_curriculum(
    request: CurriculumGenerationRequest,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Generate a curriculum using the multi-agent workflow

    This endpoint runs the complete Architect → Refiner → Verifier pipeline
    synchronously and returns the final syllabus.

    For long-running generations, use the async endpoint: POST /generate/async
    """
    start_time = datetime.utcnow()

    try:
        # Convert request to constraints
        constraints = CurriculumConstraints(
            duration_weeks=request.duration_weeks,
            difficulty_level=request.difficulty_level.value,
            target_audience=request.target_audience,
            prerequisites=request.prerequisites,
            learning_style=request.learning_style.value,
            max_modules=request.max_modules
        )

        # Initialize agent graph
        agent_graph = AgentGraph(
            graph_service=graph_service,
            config={
                "temperature": 0.7,
                "model": "gpt-4o"  # Use GPT-4o for best results
            }
        )

        # Run the workflow
        logger.info(f"Starting curriculum generation for topic: {request.topic}")
        final_state = await agent_graph.run(
            topic=request.topic,
            course_id=request.course_id,
            constraints=constraints
        )

        # Calculate generation time
        generation_time = (datetime.utcnow() - start_time).total_seconds()

        # Check for errors
        errors = final_state.get("errors", [])
        warnings = final_state.get("warnings", [])
        syllabus = final_state.get("final_syllabus")
        verification = final_state.get("verification_results", {})

        if syllabus:
            # Add generation timestamp
            syllabus["generated_at"] = datetime.utcnow().isoformat()

            return CurriculumGenerationResponse(
                success=True,
                syllabus=syllabus,
                quality_score=verification.get("quality_score", 0),
                errors=errors,
                warnings=warnings,
                generation_time_seconds=generation_time
            )
        else:
            return CurriculumGenerationResponse(
                success=False,
                errors=errors or ["Curriculum generation failed - no syllabus produced"],
                warnings=warnings,
                generation_time_seconds=generation_time
            )

    except Exception as e:
        logger.error(f"Error generating curriculum: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Curriculum generation failed: {str(e)}"
        )


@router.post("/generate/async", response_model=Dict[str, str])
async def generate_curriculum_async(
    request: CurriculumGenerationRequest,
    background_tasks: BackgroundTasks,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Start an async curriculum generation job

    Returns a job_id that can be used to check status and retrieve results.
    """
    import uuid

    job_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Initialize job
    _curriculum_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "current_agent": None,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "request": request.dict()
    }

    # Start background task
    background_tasks.add_task(
        _run_curriculum_generation_job,
        job_id,
        request,
        graph_service
    )

    return {"job_id": job_id, "status": "started"}


@router.get("/jobs/{job_id}", response_model=CurriculumJobStatus)
async def get_job_status(job_id: str):
    """
    Get the status of an async curriculum generation job
    """
    if job_id not in _curriculum_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _curriculum_jobs[job_id]
    return CurriculumJobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_agent=job.get("current_agent"),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )


@router.get("/jobs/{job_id}/result", response_model=CurriculumGenerationResponse)
async def get_job_result(job_id: str):
    """
    Get the result of a completed curriculum generation job
    """
    if job_id not in _curriculum_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _curriculum_jobs[job_id]

    if job["status"] == "pending" or job["status"] == "running":
        raise HTTPException(
            status_code=202,
            detail=f"Job still {job['status']}. Progress: {job['progress']}%"
        )

    if job["status"] == "failed":
        return CurriculumGenerationResponse(
            success=False,
            job_id=job_id,
            errors=[job.get("error", "Unknown error")]
        )

    result = job.get("result", {})
    return CurriculumGenerationResponse(
        success=True,
        job_id=job_id,
        syllabus=result.get("syllabus"),
        quality_score=result.get("quality_score"),
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
        generation_time_seconds=result.get("generation_time")
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a completed job from memory
    """
    if job_id not in _curriculum_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _curriculum_jobs[job_id]
    if job["status"] in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a job that is still running"
        )

    del _curriculum_jobs[job_id]
    return {"message": "Job deleted"}


@router.get("/preview", response_model=Dict[str, Any])
async def preview_arc_of_learning(
    topic: str,
    duration_weeks: int = 4,
    difficulty_level: DifficultyLevel = DifficultyLevel.intermediate,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Quick preview of the Arc of Learning without full curriculum generation

    This runs only the Architect agent to generate the high-level structure,
    useful for user confirmation before full generation.
    """
    from app.agents import ArchitectAgent, AgentState

    try:
        # Initialize Architect only
        architect = ArchitectAgent(graph_service=graph_service)

        # Create minimal state
        state: AgentState = {
            "topic": topic,
            "constraints": {
                "duration_weeks": duration_weeks,
                "difficulty_level": difficulty_level.value
            },
            "course_id": 0,  # Placeholder
            "messages": [],
            "knowledge_graph_data": {},
            "current_agent": "architect",
            "iteration_count": 0,
            "arc_of_learning": None,
            "learning_outcomes": None,
            "verification_results": None,
            "final_syllabus": None,
            "errors": [],
            "warnings": []
        }

        # Run architect
        result_state = await architect.process(state)

        return {
            "success": "arc_of_learning" in result_state and result_state["arc_of_learning"],
            "arc_of_learning": result_state.get("arc_of_learning"),
            "errors": result_state.get("errors", []),
            "message": "This is a preview. Use /generate to create the full curriculum."
        }

    except Exception as e:
        logger.error(f"Error in preview: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Preview failed: {str(e)}"
        )


# ============== Background Job Runner ==============

async def _run_curriculum_generation_job(
    job_id: str,
    request: CurriculumGenerationRequest,
    graph_service: AsyncGraphService
):
    """
    Background task to run curriculum generation
    """
    start_time = datetime.utcnow()

    try:
        # Update job status
        _curriculum_jobs[job_id]["status"] = "running"
        _curriculum_jobs[job_id]["updated_at"] = datetime.utcnow()
        _curriculum_jobs[job_id]["current_agent"] = "architect"
        _curriculum_jobs[job_id]["progress"] = 10

        # Convert request to constraints
        constraints = CurriculumConstraints(
            duration_weeks=request.duration_weeks,
            difficulty_level=request.difficulty_level.value,
            target_audience=request.target_audience,
            prerequisites=request.prerequisites,
            learning_style=request.learning_style.value,
            max_modules=request.max_modules
        )

        # Initialize agent graph
        agent_graph = AgentGraph(
            graph_service=graph_service,
            config={
                "temperature": 0.7,
                "model": "gpt-4o"
            }
        )

        # Run the workflow
        logger.info(f"[Job {job_id}] Starting curriculum generation")
        final_state = await agent_graph.run(
            topic=request.topic,
            course_id=request.course_id,
            constraints=constraints
        )

        # Calculate generation time
        generation_time = (datetime.utcnow() - start_time).total_seconds()

        # Update job with result
        _curriculum_jobs[job_id]["status"] = "completed"
        _curriculum_jobs[job_id]["progress"] = 100
        _curriculum_jobs[job_id]["current_agent"] = None
        _curriculum_jobs[job_id]["updated_at"] = datetime.utcnow()
        _curriculum_jobs[job_id]["result"] = {
            "syllabus": final_state.get("final_syllabus"),
            "quality_score": final_state.get("verification_results", {}).get("quality_score"),
            "errors": final_state.get("errors", []),
            "warnings": final_state.get("warnings", []),
            "generation_time": generation_time
        }

        logger.info(f"[Job {job_id}] Completed successfully in {generation_time:.1f}s")

    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {e}", exc_info=True)
        _curriculum_jobs[job_id]["status"] = "failed"
        _curriculum_jobs[job_id]["error"] = str(e)
        _curriculum_jobs[job_id]["updated_at"] = datetime.utcnow()


# ============== HiPlan Orchestrator Endpoints ==============

class HiPlanGenerationRequest(BaseModel):
    """Request for HiPlan orchestrated curriculum generation"""
    topic: str = Field(..., description="Main topic for the curriculum")
    course_id: int = Field(..., description="Course ID")
    duration_weeks: int = Field(default=4, ge=1, le=16)
    difficulty_level: DifficultyLevel = Field(default=DifficultyLevel.intermediate)
    learning_style: LearningStyle = Field(default=LearningStyle.balanced)
    max_revisions: int = Field(default=3, ge=1, le=5, description="Max revision cycles")


@router.post("/hiplan/generate", response_model=Dict[str, Any])
async def generate_with_hiplan(
    request: HiPlanGenerationRequest,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Generate curriculum using HiPlan (Hierarchical Planning) orchestrator

    This uses a more sophisticated multi-agent workflow with:
    - Hierarchical planning (global milestones -> local details)
    - Automatic revision loops when verification fails
    - Context window management for large curricula
    """
    from app.agents.hiplan_orchestrator import HiPlanOrchestrator
    from app.agents import ArchitectAgent, RefinerAgent, VerifierAgent

    try:
        # Initialize agents
        architect = ArchitectAgent(graph_service=graph_service)
        refiner = RefinerAgent(graph_service=graph_service)
        verifier = VerifierAgent(graph_service=graph_service)

        # Initialize HiPlan orchestrator
        orchestrator = HiPlanOrchestrator(
            architect_agent=architect,
            refiner_agent=refiner,
            verifier_agent=verifier,
            graph_service=graph_service
        )

        # Build constraints
        constraints = {
            "duration_weeks": request.duration_weeks,
            "difficulty_level": request.difficulty_level.value,
            "learning_style": request.learning_style.value,
            "max_revisions": request.max_revisions
        }

        # Run HiPlan workflow
        logger.info(f"Starting HiPlan generation for: {request.topic}")
        result = await orchestrator.run(
            topic=request.topic,
            constraints=constraints,
            course_id=request.course_id
        )

        return result

    except Exception as e:
        logger.error(f"HiPlan generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"HiPlan curriculum generation failed: {str(e)}"
        )


# ============== Prerequisite Extraction Endpoints ==============

class PrerequisiteAnalysisRequest(BaseModel):
    """Request to analyze prerequisite relationship between two concepts"""
    concept_a: str
    concept_b: str
    domain_context: Optional[str] = None


class ContentScanRequest(BaseModel):
    """Request to scan content for prerequisite relationships"""
    content: str = Field(..., max_length=50000)
    domain: Optional[str] = None
    known_concepts: Optional[List[str]] = None


class GapDetectionRequest(BaseModel):
    """Request to detect gaps in a syllabus"""
    syllabus: Dict[str, Any]
    course_id: int


@router.post("/prerequisites/analyze-pair", response_model=Dict[str, Any])
async def analyze_prerequisite_pair(
    request: PrerequisiteAnalysisRequest,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Analyze prerequisite relationship between two concepts

    Uses zero-shot LLM inference to determine if one concept
    requires knowledge of another.
    """
    from app.agents.prerequisite_extraction import get_prerequisite_agent

    agent = get_prerequisite_agent(graph_service)

    try:
        relation = await agent.analyze_pair(
            concept_a=request.concept_a,
            concept_b=request.concept_b,
            context=request.domain_context
        )

        if not relation:
            return {
                "has_relationship": False,
                "concept_a": request.concept_a,
                "concept_b": request.concept_b,
                "message": "No prerequisite relationship detected"
            }

        return {
            "has_relationship": True,
            "relationship": relation.to_dict()
        }

    except Exception as e:
        logger.error(f"Prerequisite analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prerequisites/scan-content", response_model=Dict[str, Any])
async def scan_content_for_prerequisites(
    request: ContentScanRequest,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Scan educational content to extract prerequisite relationships

    Analyzes text to discover:
    - Concepts mentioned
    - Prerequisite relationships between concepts
    - Can populate knowledge graph with discovered relationships
    """
    from app.agents.prerequisite_extraction import get_prerequisite_agent

    agent = get_prerequisite_agent(graph_service)

    try:
        result = await agent.scan_content(
            content=request.content,
            domain=request.domain,
            known_concepts=request.known_concepts
        )

        return {
            "concepts_discovered": result.concepts_discovered,
            "relationships": [r.to_dict() for r in result.relations],
            "relationship_count": len(result.relations),
            "metadata": result.metadata
        }

    except Exception as e:
        logger.error(f"Content scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prerequisites/detect-gaps", response_model=Dict[str, Any])
async def detect_prerequisite_gaps(
    request: GapDetectionRequest,
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Detect prerequisite gaps in a syllabus

    Analyzes a syllabus to find:
    - Missing prerequisites
    - Concepts taught out of order
    - Suggests bridge modules to fill gaps
    """
    from app.agents.prerequisite_extraction import get_prerequisite_agent

    agent = get_prerequisite_agent(graph_service)

    try:
        bridge_modules = await agent.detect_gaps(
            syllabus=request.syllabus,
            course_id=request.course_id
        )

        return {
            "gaps_found": len(bridge_modules),
            "bridge_modules": [
                {
                    "id": bm.id,
                    "title": bm.title,
                    "description": bm.description,
                    "concepts_to_teach": bm.concepts_to_teach,
                    "estimated_minutes": bm.estimated_minutes,
                    "insertion_point": bm.insertion_point,
                    "rationale": bm.rationale
                }
                for bm in bridge_modules
            ]
        }

    except Exception as e:
        logger.error(f"Gap detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prerequisites/write-to-graph", response_model=Dict[str, Any])
async def write_prerequisites_to_graph(
    course_id: int,
    relations: List[Dict[str, Any]],
    min_confidence: str = "medium",
    graph_service: AsyncGraphService = Depends(get_graph_service)
):
    """
    Write extracted prerequisite relationships to the knowledge graph

    Only writes relationships above the minimum confidence threshold.
    """
    from app.agents.prerequisite_extraction import (
        get_prerequisite_agent,
        PrerequisiteRelation,
        DependencyType,
        ConfidenceLevel
    )

    agent = get_prerequisite_agent(graph_service)

    # Convert dict relations to objects
    relation_objects = []
    for r in relations:
        try:
            rel = PrerequisiteRelation(
                source_concept=r["source"],
                prerequisite_concept=r["prerequisite"],
                dependency_type=DependencyType(r.get("type", "related")),
                confidence=ConfidenceLevel(r.get("confidence", "medium")),
                evidence=r.get("evidence", ""),
                context=r.get("context", "api_submission")
            )
            relation_objects.append(rel)
        except Exception as e:
            logger.warning(f"Skipping invalid relation: {e}")

    # Map confidence level
    confidence_map = {
        "low": ConfidenceLevel.LOW,
        "medium": ConfidenceLevel.MEDIUM,
        "high": ConfidenceLevel.HIGH
    }
    min_conf = confidence_map.get(min_confidence.lower(), ConfidenceLevel.MEDIUM)

    try:
        result = await agent.write_to_graph(
            course_id=course_id,
            relations=relation_objects,
            min_confidence=min_conf
        )

        return result

    except Exception as e:
        logger.error(f"Failed to write to graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
