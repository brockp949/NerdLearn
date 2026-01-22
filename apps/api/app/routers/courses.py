from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional
from pydantic import BaseModel
from app.core.database import get_db
from app.models.course import Course, CourseStatus, Module, ProcessingStatus
from app.schemas.course import CourseCreate, CourseUpdate, CourseResponse

router = APIRouter()


class BatchProcessingResponse(BaseModel):
    """Response for batch processing request"""
    task_id: str
    course_id: int
    module_count: int
    message: str
    mode: str  # "parallel" or "sequential"


@router.get("/", response_model=List[CourseResponse])
async def list_courses(
    status_filter: str = None,
    instructor_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all courses with optional filtering"""
    query = select(Course)

    if status_filter:
        query = query.where(Course.status == CourseStatus(status_filter))

    if instructor_id:
        query = query.where(Course.instructor_id == instructor_id)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    courses = result.scalars().all()

    return courses


@router.post("/", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
async def create_course(course: CourseCreate, db: AsyncSession = Depends(get_db)):
    """Create a new course (Instructor Studio)"""
    db_course = Course(
        title=course.title,
        description=course.description,
        instructor_id=course.instructor_id,
        thumbnail_url=course.thumbnail_url,
        price=course.price,
        difficulty_level=course.difficulty_level,
        tags=",".join(course.tags) if course.tags else "",
        status=CourseStatus.DRAFT
    )

    db.add(db_course)
    await db.flush()
    await db.refresh(db_course)

    return db_course


@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(course_id: int, db: AsyncSession = Depends(get_db)):
    """Get course by ID"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    return course


@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: int,
    course_update: CourseUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update course details"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Update fields
    update_data = course_update.dict(exclude_unset=True)
    if "tags" in update_data and update_data["tags"]:
        update_data["tags"] = ",".join(update_data["tags"])

    for field, value in update_data.items():
        setattr(course, field, value)

    await db.flush()
    await db.refresh(course)

    return course


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(course_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a course"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    await db.delete(course)

    return None


@router.post("/{course_id}/publish", response_model=CourseResponse)
async def publish_course(course_id: int, db: AsyncSession = Depends(get_db)):
    """Publish a course (change status from draft to published)"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    course.status = CourseStatus.PUBLISHED
    from datetime import datetime
    course.published_at = datetime.now()

    await db.flush()
    await db.refresh(course)

    return course


# ============== Batch Processing Endpoints ==============


@router.post("/{course_id}/process-all", response_model=BatchProcessingResponse)
async def process_all_modules(
    course_id: int,
    mode: str = Query(default="parallel", regex="^(parallel|sequential)$"),
    reprocess: bool = Query(default=False, description="Reprocess already processed modules"),
    db: AsyncSession = Depends(get_db)
):
    """
    Process all modules in a course.

    This triggers background processing for all unprocessed modules (or all modules
    if reprocess=True).

    Args:
        course_id: Course ID
        mode: Processing mode - "parallel" (faster) or "sequential" (less resource intensive)
        reprocess: If True, reprocess even already processed modules

    Returns:
        Task ID and processing details
    """
    # Get course with modules
    result = await db.execute(
        select(Course)
        .options(selectinload(Course.modules))
        .where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Filter modules to process
    modules_to_process = []
    for module in course.modules:
        # Skip if already processed and not reprocessing
        if not reprocess and module.processing_status == ProcessingStatus.COMPLETED:
            continue

        # Skip if currently processing
        if module.processing_status == ProcessingStatus.PROCESSING:
            continue

        # Only process PDF and video modules
        if module.module_type.value not in ("pdf", "video"):
            continue

        if not module.file_url:
            continue

        modules_to_process.append({
            "id": module.id,
            "type": module.module_type.value,
            "file_path": module.file_url,
            "title": module.title,
        })

    if not modules_to_process:
        raise HTTPException(
            status_code=400,
            detail="No modules to process. All modules may already be processed."
        )

    # Trigger batch processing task
    try:
        from app.services.worker_client import celery_app

        if mode == "parallel":
            task = celery_app.send_task(
                "process_course_batch",
                args=[course_id, modules_to_process]
            )
        else:
            task = celery_app.send_task(
                "process_course_sequential",
                args=[course_id, modules_to_process]
            )

        # Update module statuses
        for module in course.modules:
            if any(m["id"] == module.id for m in modules_to_process):
                module.processing_status = ProcessingStatus.PROCESSING
                module.processing_task_id = task.id

        await db.commit()

        return BatchProcessingResponse(
            task_id=task.id,
            course_id=course_id,
            module_count=len(modules_to_process),
            message=f"Started {mode} processing for {len(modules_to_process)} modules",
            mode=mode,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch processing: {str(e)}"
        )


@router.get("/{course_id}/processing-status")
async def get_course_processing_status(
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get processing status for all modules in a course.

    Returns detailed status for each module including progress and errors.
    """
    result = await db.execute(
        select(Course)
        .options(selectinload(Course.modules))
        .where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    modules_status = []
    for module in course.modules:
        modules_status.append({
            "module_id": module.id,
            "title": module.title,
            "module_type": module.module_type.value,
            "processing_status": module.processing_status.value,
            "task_id": module.processing_task_id,
            "is_processed": module.is_processed,
            "chunk_count": module.chunk_count,
            "concept_count": module.concept_count,
            "processed_at": module.processed_at.isoformat() if module.processed_at else None,
            "error": module.processing_error,
        })

    # Calculate summary
    total = len(modules_status)
    completed = sum(1 for m in modules_status if m["processing_status"] == "completed")
    processing = sum(1 for m in modules_status if m["processing_status"] == "processing")
    failed = sum(1 for m in modules_status if m["processing_status"] == "failed")
    pending = sum(1 for m in modules_status if m["processing_status"] == "pending")

    return {
        "course_id": course_id,
        "course_title": course.title,
        "summary": {
            "total": total,
            "completed": completed,
            "processing": processing,
            "failed": failed,
            "pending": pending,
            "progress_percentage": (completed / total * 100) if total > 0 else 0,
        },
        "modules": modules_status,
    }
