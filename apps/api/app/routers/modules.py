from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.core.database import get_db
from app.models.course import Module, Course, ModuleType, ProcessingStatus
from app.schemas.course import ModuleCreate, ModuleResponse, ModuleUpdate
from app.services.storage import storage_service
from app.services.worker_client import trigger_module_processing
import os

router = APIRouter()


@router.get("/courses/{course_id}/modules", response_model=List[ModuleResponse])
async def list_modules(course_id: int, db: AsyncSession = Depends(get_db)):
    """List all modules for a course"""
    result = await db.execute(
        select(Module).where(Module.course_id == course_id).order_by(Module.order)
    )
    modules = result.scalars().all()
    return modules


@router.post("/courses/{course_id}/modules", response_model=ModuleResponse, status_code=status.HTTP_201_CREATED)
async def create_module(
    course_id: int,
    title: str = Form(...),
    description: str = Form(None),
    module_type: str = Form(...),
    order: int = Form(...),
    duration_minutes: int = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload a module (video/PDF) to a course"""
    # Verify course exists
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Upload file to MinIO/S3
    file_extension = os.path.splitext(file.filename)[1]
    file_key = f"courses/{course_id}/modules/{title.replace(' ', '_')}{file_extension}"

    try:
        file_url = await storage_service.upload_file(
            file.file,
            file_key,
            file.content_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    # Create module record
    db_module = Module(
        course_id=course_id,
        title=title,
        description=description,
        module_type=ModuleType(module_type),
        order=order,
        duration_minutes=duration_minutes,
        file_url=file_url,
        file_size=file.size if hasattr(file, 'size') else None
    )

    db.add(db_module)
    await db.flush()
    await db.refresh(db_module)

    # Trigger background processing task
    try:
        task_id = trigger_module_processing(
            module_id=db_module.id,
            course_id=course_id,
            module_type=module_type,
            file_url=file_url,
            title=title
        )
        db_module.processing_task_id = task_id
        db_module.processing_status = ProcessingStatus.PROCESSING
        await db.commit()
    except Exception as e:
        # Log error but don't fail the request
        print(f"Failed to trigger processing task: {str(e)}")
        db_module.processing_status = ProcessingStatus.FAILED
        db_module.processing_error = str(e)
        await db.commit()

    return db_module


@router.get("/modules/{module_id}", response_model=ModuleResponse)
async def get_module(module_id: int, db: AsyncSession = Depends(get_db)):
    """Get module by ID"""
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    return module


@router.put("/modules/{module_id}", response_model=ModuleResponse)
async def update_module(
    module_id: int,
    module_update: ModuleUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update module details"""
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Update fields
    update_data = module_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(module, field, value)

    await db.flush()
    await db.refresh(module)

    return module


@router.delete("/modules/{module_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_module(module_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a module"""
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # TODO: Delete file from storage
    # await storage_service.delete_file(module.file_url)

    await db.delete(module)

    return None


# ============== Prerequisite Management Endpoints ==============


@router.get("/modules/{module_id}/prerequisites")
async def get_prerequisites(module_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get all prerequisites for a module.

    Returns a list of modules that must be completed before this module.
    """
    result = await db.execute(
        select(Module).where(Module.id == module_id)
    )
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Load prerequisites
    await db.refresh(module, ["prerequisites"])

    return {
        "module_id": module_id,
        "module_title": module.title,
        "prerequisites": [
            {
                "id": prereq.id,
                "title": prereq.title,
                "order": prereq.order,
                "module_type": prereq.module_type.value,
            }
            for prereq in module.prerequisites
        ],
    }


@router.post("/modules/{module_id}/prerequisites/{prerequisite_id}")
async def add_prerequisite(
    module_id: int,
    prerequisite_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Add a prerequisite to a module.

    The prerequisite module must be completed before the target module
    can be accessed by learners.
    """
    if module_id == prerequisite_id:
        raise HTTPException(
            status_code=400,
            detail="A module cannot be its own prerequisite"
        )

    # Get both modules
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    result = await db.execute(select(Module).where(Module.id == prerequisite_id))
    prerequisite = result.scalar_one_or_none()

    if not prerequisite:
        raise HTTPException(status_code=404, detail="Prerequisite module not found")

    # Verify both modules are in the same course
    if module.course_id != prerequisite.course_id:
        raise HTTPException(
            status_code=400,
            detail="Prerequisite must be from the same course"
        )

    # Load current prerequisites
    await db.refresh(module, ["prerequisites"])

    # Check for circular dependency
    if await _has_circular_dependency(db, module_id, prerequisite_id):
        raise HTTPException(
            status_code=400,
            detail="Adding this prerequisite would create a circular dependency"
        )

    # Check if already a prerequisite
    if prerequisite in module.prerequisites:
        raise HTTPException(
            status_code=400,
            detail="This module is already a prerequisite"
        )

    # Add prerequisite
    module.prerequisites.append(prerequisite)
    await db.commit()

    return {
        "message": "Prerequisite added successfully",
        "module_id": module_id,
        "prerequisite_id": prerequisite_id,
    }


@router.delete("/modules/{module_id}/prerequisites/{prerequisite_id}")
async def remove_prerequisite(
    module_id: int,
    prerequisite_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Remove a prerequisite from a module.
    """
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Load prerequisites
    await db.refresh(module, ["prerequisites"])

    # Find and remove the prerequisite
    prereq_to_remove = None
    for prereq in module.prerequisites:
        if prereq.id == prerequisite_id:
            prereq_to_remove = prereq
            break

    if not prereq_to_remove:
        raise HTTPException(
            status_code=404,
            detail="Prerequisite not found for this module"
        )

    module.prerequisites.remove(prereq_to_remove)
    await db.commit()

    return {
        "message": "Prerequisite removed successfully",
        "module_id": module_id,
        "prerequisite_id": prerequisite_id,
    }


@router.put("/modules/{module_id}/prerequisites")
async def set_prerequisites(
    module_id: int,
    prerequisite_ids: List[int],
    db: AsyncSession = Depends(get_db)
):
    """
    Set all prerequisites for a module (replaces existing).

    Validates for circular dependencies before applying.
    """
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Validate no self-reference
    if module_id in prerequisite_ids:
        raise HTTPException(
            status_code=400,
            detail="A module cannot be its own prerequisite"
        )

    # Get all prerequisite modules
    result = await db.execute(
        select(Module).where(Module.id.in_(prerequisite_ids))
    )
    prerequisites = result.scalars().all()

    if len(prerequisites) != len(prerequisite_ids):
        raise HTTPException(
            status_code=404,
            detail="One or more prerequisite modules not found"
        )

    # Verify all are from the same course
    for prereq in prerequisites:
        if prereq.course_id != module.course_id:
            raise HTTPException(
                status_code=400,
                detail=f"Prerequisite {prereq.id} is from a different course"
            )

    # Check for circular dependencies
    for prereq_id in prerequisite_ids:
        if await _has_circular_dependency(db, module_id, prereq_id):
            raise HTTPException(
                status_code=400,
                detail=f"Adding prerequisite {prereq_id} would create a circular dependency"
            )

    # Replace prerequisites
    await db.refresh(module, ["prerequisites"])
    module.prerequisites = list(prerequisites)
    await db.commit()

    return {
        "message": "Prerequisites updated successfully",
        "module_id": module_id,
        "prerequisite_ids": prerequisite_ids,
    }


async def _has_circular_dependency(
    db: AsyncSession,
    module_id: int,
    new_prereq_id: int,
    visited: set = None
) -> bool:
    """
    Check if adding new_prereq_id as a prerequisite to module_id
    would create a circular dependency.

    Uses DFS to detect cycles in the prerequisite graph.
    """
    if visited is None:
        visited = set()

    if new_prereq_id in visited:
        return False  # Already checked this path

    visited.add(new_prereq_id)

    # Get the new prerequisite's prerequisites
    result = await db.execute(
        select(Module).where(Module.id == new_prereq_id)
    )
    prereq_module = result.scalar_one_or_none()

    if not prereq_module:
        return False

    await db.refresh(prereq_module, ["prerequisites"])

    # Check if module_id is in the prerequisite chain
    for prereq in prereq_module.prerequisites:
        if prereq.id == module_id:
            return True  # Circular dependency found
        if await _has_circular_dependency(db, module_id, prereq.id, visited):
            return True

    return False
