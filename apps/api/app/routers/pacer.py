"""
PACER Learning Protocol API Endpoints
Content classification, analogy critique, evidence linking, and procedural tracking

P.A.C.E.R. Framework:
- P: Procedural (Practice) - How-to steps -> Practice execution
- A: Analogous (Critique) - Metaphors -> Breakdown identification
- C: Conceptual (Mapping) - Theories -> Knowledge graph mapping
- E: Evidence (Store & Rehearse) - Facts -> Link to concepts
- R: Reference (Store & Rehearse) - Details -> SRS flashcards
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib

from app.core.database import get_db
from app.models.pacer import (
    PACERType,
    PACERContentItem,
    AnalogyRecord,
    EvidenceConceptLink,
    ProceduralProgress,
    AnalogyCritique,
    UserPACERProfile,
    PACERClassificationLog,
    EvidenceRelationshipType,
)
from app.models.spaced_repetition import Concept
from app.adaptive.pacer import (
    PACERClassifier,
    AnalogyCritiqueEngine,
    EvidenceLinkingService,
    ProceduralProgressTracker,
)

router = APIRouter(prefix="/pacer", tags=["PACER Protocol"])

# Initialize services
classifier = PACERClassifier(use_llm=False)


# === Request/Response Models ===


class ClassifyRequest(BaseModel):
    """Request to classify content into PACER type"""

    content: str = Field(..., min_length=10, description="Content to classify")
    context: Optional[dict] = Field(None, description="Optional context for classification")


class ClassifyResponse(BaseModel):
    """Classification result"""

    pacer_type: str
    confidence: float
    reasoning: str
    alternatives: List[dict]
    recommended_action: dict
    content_hash: str


class TriageStepResponse(BaseModel):
    """Single step in the triage decision tree"""

    question: str
    answer: bool
    leads_to: Optional[str]


class ContentCreateRequest(BaseModel):
    """Request to create PACER-tagged content"""

    course_id: int
    module_id: Optional[int] = None
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=10)
    pacer_type: Optional[str] = None
    metadata: dict = {}


class ContentResponse(BaseModel):
    """PACER content item response"""

    id: int
    course_id: int
    module_id: Optional[int]
    pacer_type: str
    title: str
    content: str
    classification_confidence: float
    metadata: dict = Field(default_factory=dict)
    created_at: datetime


class AnalogyCreateRequest(BaseModel):
    """Request to create an analogy"""

    course_id: int
    module_id: Optional[int] = None
    title: str
    content: str
    source_domain: str = Field(..., description="Familiar concept (e.g., 'Water Flow')")
    target_domain: str = Field(..., description="New concept (e.g., 'Electricity')")
    mappings: List[dict] = Field(
        ..., description="Element mappings between domains"
    )
    breakdown_points: List[dict] = Field(
        ..., description="Where the analogy fails"
    )


class CritiqueSubmitRequest(BaseModel):
    """Request to submit analogy critique"""

    identified_breakdowns: List[str] = Field(
        ..., description="Aspects user identified as breakdowns"
    )
    explanations: List[str] = Field(
        default=[], description="Explanations for each breakdown"
    )


class CritiqueResponse(BaseModel):
    """Critique evaluation result"""

    score: float
    precision: float
    recall: float
    correctly_identified: List[str]
    missed_breakdowns: List[dict]
    false_positives: List[str]
    feedback: str


class EvidenceLinkRequest(BaseModel):
    """Request to link evidence to concepts"""

    evidence_item_id: int
    concept_ids: List[int]
    relationship_type: str = "supports"
    strength: float = Field(0.7, ge=0.0, le=1.0)
    citation: Optional[str] = None


class ProceduralStepRequest(BaseModel):
    """Request to complete a procedural step"""

    step_number: int = Field(..., ge=0)
    success: bool
    time_ms: int = Field(..., ge=0)
    error_count: int = Field(0, ge=0)


class ProceduralStatusResponse(BaseModel):
    """Procedural progress status"""

    item_id: int
    title: str
    current_step: int
    total_steps: int
    completed: bool
    attempts: int
    progress_percent: float


class UserProfileResponse(BaseModel):
    """User PACER profile"""

    user_id: int
    procedural_proficiency: float
    analogous_proficiency: float
    conceptual_proficiency: float
    evidence_proficiency: float
    reference_proficiency: float
    total_items_processed: int
    preferred_types: List[str]


# === Classification Endpoints ===


@router.post("/classify", response_model=ClassifyResponse)
async def classify_content(
    request: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Classify content into PACER type using AI-powered triage.

    Returns the classification with confidence score, reasoning,
    and recommended learning action.
    """
    result = classifier.classify(request.content, request.context)
    recommended_action = classifier.get_recommended_action(result.pacer_type)

    # Log classification
    log = PACERClassificationLog(
        content_hash=result.content_hash,
        content_preview=request.content[:500],
        predicted_type=result.pacer_type,
        confidence=result.confidence,
        alternative_types=[
            {"type": t.value, "confidence": c} for t, c in result.alternative_types
        ],
        method=result.metadata.get("method", "rule_based"),
    )
    db.add(log)

    return ClassifyResponse(
        pacer_type=result.pacer_type.value,
        confidence=result.confidence,
        reasoning=result.reasoning,
        alternatives=[
            {"type": t.value, "confidence": c} for t, c in result.alternative_types
        ],
        recommended_action=recommended_action,
        content_hash=result.content_hash,
    )


@router.post("/classify/triage", response_model=List[TriageStepResponse])
async def run_triage_tree(
    request: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run interactive triage decision tree.

    Returns the decision path taken through the PACER triage logic,
    showing which questions led to the final classification.
    """
    decisions = classifier.run_triage_tree(request.content)

    return [
        TriageStepResponse(
            question=d.question,
            answer=d.answer,
            leads_to=d.leads_to.value if d.leads_to else None,
        )
        for d in decisions
    ]


@router.post("/classify/batch", response_model=List[ClassifyResponse])
async def classify_batch(
    contents: List[str] = Field(..., min_items=1, max_items=50),
    context: Optional[dict] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Classify multiple content items in batch.
    Efficient for processing course content.
    """
    results = classifier.classify_batch(contents, context)

    return [
        ClassifyResponse(
            pacer_type=r.pacer_type.value,
            confidence=r.confidence,
            reasoning=r.reasoning,
            alternatives=[
                {"type": t.value, "confidence": c} for t, c in r.alternative_types
            ],
            recommended_action=classifier.get_recommended_action(r.pacer_type),
            content_hash=r.content_hash,
        )
        for r in results
    ]


# === Content Management Endpoints ===


@router.post("/content", response_model=ContentResponse)
async def create_pacer_content(
    request: ContentCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Create PACER-classified content item.
    Auto-classifies if pacer_type not provided.
    """
    # Auto-classify if type not provided
    if not request.pacer_type:
        result = classifier.classify(request.content)
        pacer_type = result.pacer_type
        confidence = result.confidence
    else:
        pacer_type = PACERType(request.pacer_type)
        confidence = 1.0

    # Generate content hash
    content_hash = hashlib.sha256(request.content.encode()).hexdigest()[:16]

    # Create content item
    item = PACERContentItem(
        course_id=request.course_id,
        module_id=request.module_id,
        pacer_type=pacer_type,
        title=request.title,
        content=request.content,
        content_hash=content_hash,
        classification_confidence=confidence,
        classification_confidence=confidence,
        classification_method="manual" if request.pacer_type else "ai_triage",
        item_metadata=request.metadata,
    )

    db.add(item)
    await db.flush()

    return ContentResponse(
        id=item.id,
        course_id=item.course_id,
        module_id=item.module_id,
        pacer_type=item.pacer_type.value,
        title=item.title,
        content=item.content,
        classification_confidence=item.classification_confidence,
        metadata=item.item_metadata or {},
        created_at=item.created_at or datetime.utcnow(),
    )


@router.get("/content/{course_id}", response_model=List[ContentResponse])
async def get_pacer_content(
    course_id: int,
    pacer_type: Optional[str] = Query(None, description="Filter by PACER type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Get PACER content for a course.
    Optionally filter by PACER type.
    """
    query = select(PACERContentItem).where(PACERContentItem.course_id == course_id)

    if pacer_type:
        query = query.where(PACERContentItem.pacer_type == PACERType(pacer_type))

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    items = result.scalars().all()

    return [
        ContentResponse(
            id=item.id,
            course_id=item.course_id,
            module_id=item.module_id,
            pacer_type=item.pacer_type.value,
            title=item.title,
            content=item.content,
            classification_confidence=item.classification_confidence,
            metadata=item.item_metadata or {},
            created_at=item.created_at or datetime.utcnow(),
        )
        for item in items
    ]


# === Analogy Endpoints ===


@router.post("/analogies")
async def create_analogy(
    request: AnalogyCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Create an analogy with breakdown points for critique learning.
    """
    # Create PACER content item
    content_hash = hashlib.sha256(request.content.encode()).hexdigest()[:16]

    item = PACERContentItem(
        course_id=request.course_id,
        module_id=request.module_id,
        pacer_type=PACERType.ANALOGOUS,
        title=request.title,
        content=request.content,
        content_hash=content_hash,
        classification_confidence=1.0,
        classification_confidence=1.0,
        classification_method="manual",
        item_metadata={
            "source_domain": request.source_domain,
            "target_domain": request.target_domain,
        },
    )
    db.add(item)
    await db.flush()

    # Create analogy engine and save
    engine = AnalogyCritiqueEngine(db)
    analogy = engine.create_analogy(
        source_domain=request.source_domain,
        target_domain=request.target_domain,
        content=request.content,
        mappings=request.mappings,
        breakdowns=request.breakdown_points,
    )

    record = await engine.save_analogy_to_db(analogy, item.id)

    return {
        "id": record.id,
        "pacer_item_id": item.id,
        "source_domain": record.source_domain,
        "target_domain": record.target_domain,
        "mappings": record.structural_mapping,
        "breakdown_count": len(record.breakdown_points),
        "critique_prompt": record.critique_prompt,
    }


@router.get("/analogies/{analogy_id}")
async def get_analogy(
    analogy_id: int,
    include_breakdowns: bool = Query(False, description="Include breakdown points (for instructors)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get an analogy for critique practice.
    Breakdown points hidden by default (revealed after critique).
    """
    result = await db.execute(
        select(AnalogyRecord).where(AnalogyRecord.id == analogy_id)
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Analogy not found")

    response = {
        "id": record.id,
        "source_domain": record.source_domain,
        "target_domain": record.target_domain,
        "mappings": record.structural_mapping,
        "valid_aspects": record.valid_aspects,
        "critique_prompt": record.critique_prompt,
    }

    if include_breakdowns:
        response["breakdown_points"] = record.breakdown_points

    return response


@router.post("/analogies/{analogy_id}/critique", response_model=CritiqueResponse)
async def submit_analogy_critique(
    analogy_id: int,
    request: CritiqueSubmitRequest,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit user's critique of an analogy.
    Returns evaluation with score and reveals missed breakdown points.
    """
    engine = AnalogyCritiqueEngine(db)
    analogy = await engine.get_analogy_by_id(analogy_id)

    if not analogy:
        raise HTTPException(status_code=404, detail="Analogy not found")

    # Evaluate critique with fuzzy matching
    evaluation = engine.evaluate_critique_fuzzy(
        analogy,
        request.identified_breakdowns,
        similarity_threshold=0.5,
    )

    # Save critique to database
    await engine.save_critique_to_db(
        user_id=user_id,
        analogy_id=analogy_id,
        identified_breakdowns=[
            {"aspect": b, "explanation": e}
            for b, e in zip(
                request.identified_breakdowns,
                request.explanations + [""] * (len(request.identified_breakdowns) - len(request.explanations)),
            )
        ],
        evaluation=evaluation,
    )

    # Update user profile
    await _update_user_pacer_profile(db, user_id, PACERType.ANALOGOUS, evaluation.score)

    return CritiqueResponse(
        score=evaluation.score,
        precision=evaluation.precision,
        recall=evaluation.recall,
        correctly_identified=evaluation.correctly_identified,
        missed_breakdowns=evaluation.missed_breakdowns,
        false_positives=evaluation.false_positives,
        feedback=evaluation.feedback,
    )


# === Evidence Linking Endpoints ===


@router.post("/evidence/link")
async def link_evidence_to_concepts(
    request: EvidenceLinkRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Link evidence content to concepts it supports/contradicts/qualifies.
    """
    service = EvidenceLinkingService(db)

    # Validate evidence item exists and is Evidence type
    result = await db.execute(
        select(PACERContentItem).where(PACERContentItem.id == request.evidence_item_id)
    )
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="Evidence item not found")

    if item.pacer_type != PACERType.EVIDENCE:
        raise HTTPException(
            status_code=400,
            detail=f"Item is {item.pacer_type.value}, not evidence",
        )

    # Create links
    links = await service.link_evidence_to_concepts(
        evidence_item_id=request.evidence_item_id,
        concept_ids=request.concept_ids,
        relationship_type=EvidenceRelationshipType(request.relationship_type),
        strength=request.strength,
        citation=request.citation,
    )

    return {
        "evidence_item_id": request.evidence_item_id,
        "links_created": len(links),
        "concept_ids": request.concept_ids,
    }


@router.get("/evidence/concept/{concept_id}")
async def get_evidence_for_concept(
    concept_id: int,
    relationship_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Get all evidence linked to a concept.
    Useful for study: "What proves this theory?"
    """
    service = EvidenceLinkingService(db)

    rel_type = EvidenceRelationshipType(relationship_type) if relationship_type else None
    evidence = await service.get_evidence_for_concept(concept_id, rel_type)

    return {
        "concept_id": concept_id,
        "evidence_count": len(evidence),
        "evidence": evidence,
    }


@router.post("/evidence/auto-link")
async def auto_link_evidence(
    evidence_content: str,
    course_id: int,
    min_relevance: float = Query(0.3, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    """
    Automatically suggest concept links for evidence content.
    """
    # Get concepts for course
    result = await db.execute(
        select(Concept).where(Concept.course_id == course_id)
    )
    concepts = result.scalars().all()

    candidate_concepts = [
        {"id": c.id, "name": c.name, "description": c.description or ""}
        for c in concepts
    ]

    service = EvidenceLinkingService(db)
    suggestions = await service.auto_link_evidence(
        evidence_content, candidate_concepts, min_relevance
    )

    return {
        "suggestions": suggestions,
        "total_concepts_checked": len(candidate_concepts),
    }


# === Procedural Progress Endpoints ===


@router.post("/procedural/{item_id}/start")
async def start_procedure(
    item_id: int,
    user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Start or restart a procedural learning item.
    """
    # Get item and validate
    result = await db.execute(
        select(PACERContentItem).where(PACERContentItem.id == item_id)
    )
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if item.pacer_type != PACERType.PROCEDURAL:
        raise HTTPException(
            status_code=400,
            detail=f"Item is {item.pacer_type.value}, not procedural",
        )

    # Get total steps from metadata
    total_steps = (item.item_metadata or {}).get("total_steps", 1)

    tracker = ProceduralProgressTracker(db)
    progress = await tracker.start_procedure(user_id, item_id, total_steps)

    return {
        "item_id": item_id,
        "total_steps": total_steps,
        "attempt": progress.attempts,
        "status": "started",
    }


@router.post("/procedural/{item_id}/step")
async def complete_procedural_step(
    item_id: int,
    request: ProceduralStepRequest,
    user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Complete a step in procedural learning.
    """
    tracker = ProceduralProgressTracker(db)

    try:
        result = await tracker.complete_step(
            user_id=user_id,
            pacer_item_id=item_id,
            step_number=request.step_number,
            success=request.success,
            time_ms=request.time_ms,
            error_count=request.error_count,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update user profile
    if result.success:
        await _update_user_pacer_profile(
            db, user_id, PACERType.PROCEDURAL, 1.0 if result.error_count == 0 else 0.7
        )

    return {
        "step_number": result.step_number,
        "success": result.success,
        "time_ms": result.time_ms,
        "error_count": result.error_count,
        "feedback": result.feedback,
    }


@router.get("/procedural/{item_id}/status", response_model=ProceduralStatusResponse)
async def get_procedural_status(
    item_id: int,
    user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's current progress on a procedural item.
    """
    tracker = ProceduralProgressTracker(db)
    status = await tracker.get_progress_status(user_id, item_id)

    if not status:
        raise HTTPException(status_code=404, detail="Progress not found")

    return ProceduralStatusResponse(
        item_id=status.item_id,
        title=status.title,
        current_step=status.current_step,
        total_steps=status.total_steps,
        completed=status.completed,
        attempts=status.attempts,
        progress_percent=status.progress_percent,
    )


@router.get("/procedural/user/{user_id}/active")
async def get_user_active_procedures(
    user_id: int,
    include_completed: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all procedures a user is working on.
    """
    tracker = ProceduralProgressTracker(db)
    statuses = await tracker.get_user_active_procedures(user_id, include_completed)

    return {
        "user_id": user_id,
        "procedures": [
            {
                "item_id": s.item_id,
                "title": s.title,
                "current_step": s.current_step,
                "total_steps": s.total_steps,
                "completed": s.completed,
                "progress_percent": s.progress_percent,
            }
            for s in statuses
        ],
    }


# === User Profile Endpoints ===


@router.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_user_pacer_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's PACER learning profile.
    Shows proficiency across all types.
    """
    result = await db.execute(
        select(UserPACERProfile).where(UserPACERProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()

    if not profile:
        # Create default profile
        profile = UserPACERProfile(user_id=user_id)
        db.add(profile)
        await db.flush()

    return UserProfileResponse(
        user_id=profile.user_id,
        procedural_proficiency=profile.procedural_proficiency,
        analogous_proficiency=profile.analogous_proficiency,
        conceptual_proficiency=profile.conceptual_proficiency,
        evidence_proficiency=profile.evidence_proficiency,
        reference_proficiency=profile.reference_proficiency,
        total_items_processed=profile.total_items_processed,
        preferred_types=profile.preferred_types or [],
    )


# === Helper Functions ===


async def _update_user_pacer_profile(
    db: AsyncSession,
    user_id: int,
    pacer_type: PACERType,
    performance_score: float,
):
    """Update user's PACER profile based on performance"""
    result = await db.execute(
        select(UserPACERProfile).where(UserPACERProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()

    if not profile:
        profile = UserPACERProfile(user_id=user_id)
        db.add(profile)

    # Update proficiency with exponential moving average
    alpha = 0.3  # Learning rate
    current = getattr(profile, f"{pacer_type.value}_proficiency", 0.5)
    new_value = alpha * performance_score + (1 - alpha) * current
    setattr(profile, f"{pacer_type.value}_proficiency", new_value)

    # Update item count
    profile.total_items_processed = (profile.total_items_processed or 0) + 1
    items_by_type = profile.items_by_type or {}
    items_by_type[pacer_type.value] = items_by_type.get(pacer_type.value, 0) + 1
    profile.items_by_type = items_by_type

    await db.flush()
