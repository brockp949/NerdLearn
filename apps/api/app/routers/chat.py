from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter()


@router.post("/")
async def chat_with_content(db: AsyncSession = Depends(get_db)):
    """Chat with course content (NotebookLM-style)"""
    return {"message": "Chat endpoint - to be implemented"}
