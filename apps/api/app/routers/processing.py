"""
Processing Status Router
Check status of background processing tasks
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.course import Module
from app.services.worker_client import get_task_status
from typing import Dict, Any

router = APIRouter()


@router.get("/modules/{module_id}/processing-status")
async def get_module_processing_status(
    module_id: int, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get processing status for a module"""
    # Get module
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Get task status if task exists
    task_status = None
    if module.processing_task_id:
        task_status = get_task_status(module.processing_task_id)

    return {
        "module_id": module_id,
        "processing_status": module.processing_status.value if module.processing_status else "pending",
        "is_processed": module.is_processed,
        "task_id": module.processing_task_id,
        "task_status": task_status,
        "error": module.processing_error,
        "chunk_count": module.chunk_count,
        "concept_count": module.concept_count,
        "processed_at": module.processed_at,
    }
