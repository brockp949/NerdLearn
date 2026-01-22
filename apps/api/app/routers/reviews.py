from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter()


@router.get("/due")
async def get_due_reviews(db: AsyncSession = Depends(get_db)):
    """Get items due for spaced repetition review"""
    return {"reviews": []}
