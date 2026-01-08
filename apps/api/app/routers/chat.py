from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.core.database import get_db
from app.models.gamification import ChatHistory
from app.models.assessment import UserConceptMastery
from app.chat import RAGChatEngine, ChatMessage
from app.config import settings
from pydantic import BaseModel
from typing import Optional, List, Dict
import os

router = APIRouter()

# Initialize RAG chat engine
rag_engine = RAGChatEngine(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    model="gpt-4"
)


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    user_id: int
    course_id: int
    session_id: Optional[str] = None
    module_id: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    citations: List[Dict]
    xp_earned: int = 5


@router.post("/", response_model=ChatResponse)
async def chat_with_content(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with course content using RAG

    Returns contextualized responses with citations
    """
    from app.worker.app.services.vector_store import VectorStoreService
    from app.gamification import GamificationEngine

    try:
        # Get vector store service
        vector_store = VectorStoreService()

        # Get user mastery levels for adaptive responses
        mastery_result = await db.execute(
            select(UserConceptMastery).where(UserConceptMastery.user_id == request.user_id)
        )
        masteries = mastery_result.scalars().all()
        user_mastery = {m.concept_id: m.mastery_level for m in masteries}

        # Get chat response with RAG
        response = await rag_engine.chat(
            query=request.query,
            user_id=request.user_id,
            course_id=request.course_id,
            vector_store_service=vector_store,
            session_id=request.session_id,
            module_id=request.module_id,
            user_mastery=user_mastery,
        )

        # Save to chat history
        user_message = ChatHistory(
            user_id=request.user_id,
            course_id=request.course_id,
            session_id=request.session_id or "default",
            role="user",
            content=request.query,
            citations=[],
            concept_ids=[],
        )
        db.add(user_message)

        assistant_message = ChatHistory(
            user_id=request.user_id,
            course_id=request.course_id,
            session_id=request.session_id or "default",
            role="assistant",
            content=response.content,
            citations=[c.dict() for c in response.citations],
            concept_ids=response.concept_ids,
        )
        db.add(assistant_message)

        # Award XP for chat interaction
        xp_earned = GamificationEngine.award_xp("chat_interaction")

        await db.commit()

        return ChatResponse(
            message=response.content,
            citations=[c.dict() for c in response.citations],
            xp_earned=xp_earned,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/history")
async def get_chat_history(
    user_id: int,
    course_id: int,
    session_id: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get chat history for a session"""
    session_filter = session_id or "default"

    result = await db.execute(
        select(ChatHistory)
        .where(
            and_(
                ChatHistory.user_id == user_id,
                ChatHistory.course_id == course_id,
                ChatHistory.session_id == session_filter,
            )
        )
        .order_by(ChatHistory.timestamp.desc())
        .limit(limit)
    )

    messages = result.scalars().all()

    return {
        "user_id": user_id,
        "course_id": course_id,
        "session_id": session_filter,
        "message_count": len(messages),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "citations": msg.citations,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in reversed(messages)
        ],
    }


@router.delete("/history")
async def clear_chat_history(
    user_id: int,
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Clear chat history for a session"""
    session_filter = session_id or "default"

    # Clear from database
    result = await db.execute(
        select(ChatHistory).where(
            and_(
                ChatHistory.user_id == user_id,
                ChatHistory.session_id == session_filter,
            )
        )
    )
    messages = result.scalars().all()

    for msg in messages:
        await db.delete(msg)

    await db.commit()

    # Clear from RAG engine memory
    rag_engine.clear_history(user_id, session_filter)

    return {
        "message": "Chat history cleared",
        "messages_deleted": len(messages),
    }
