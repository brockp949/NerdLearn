from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.core.database import get_db
from app.models.course import Module, Course, ModuleType
from app.schemas.course import ModuleCreate, ModuleResponse, ModuleUpdate
from app.services.storage import storage_service
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

    # TODO: Trigger background processing task for video transcription/PDF parsing

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
