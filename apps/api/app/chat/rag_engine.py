"""
RAG-based Chat System
Context-aware conversational AI using vector search and knowledge graphs
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import openai
from app.core.config import settings


class Citation(BaseModel):
    """Citation reference to source material"""
    module_id: int
    module_title: str
    module_type: str  # pdf, video
    chunk_text: str
    page_number: Optional[int] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    relevance_score: float


class ChatMessage(BaseModel):
    """Chat message with context"""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = datetime.now()
    citations: List[Citation] = []
    concept_ids: List[int] = []


class RAGChatEngine:
    """
    Retrieval-Augmented Generation chat engine
    Uses vector search + knowledge graph for context-aware responses
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        """
        Initialize RAG chat engine

        Args:
            openai_api_key: OpenAI API key
            model: Model to use (gpt-4, gpt-3.5-turbo)
        """
        self.model = model
        openai.api_key = openai_api_key
        self.conversation_history: Dict[str, List[ChatMessage]] = {}

    async def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document based on the query (HyDE)
        """
        messages = [
            {"role": "system", "content": "You are a teacher writing a brief, factual paragraph to answer a student's question. Use placeholder terms if details are unknown."},
            {"role": "user", "content": f"Write a one-paragraph factual answer to this question: {query}"}
        ]
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo", # Use a smaller model for speed
                messages=messages,
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content
        except:
            return query # Fallback to original query on error

    async def retrieve_context(
        self,
        query: str,
        course_id: int,
        vector_store_service,
        top_k: int = 5,
        module_id: Optional[int] = None,
        use_hyde: bool = True
    ) -> List[Citation]:
        """
        Retrieve relevant context from vector store with optional HyDE expansion
        """
        search_query = query
        
        if use_hyde:
            # Expand query using HyDE
            hypothetical_doc = await self._generate_hypothetical_document(query)
            # Combine original query with hypothetical doc for expansion
            search_query = f"{query}\n{hypothetical_doc}"

        # Search vector store
        results = vector_store_service.search(
            query=search_query,
            course_id=course_id,
            module_id=module_id,
            limit=top_k,
        )

        # Convert to citations
        citations = []
        for result in results:
            citation = Citation(
                module_id=result["module_id"],
                module_title=result.get("module_title", f"Module {result['module_id']}"),
                module_type=result.get("module_type", "unknown"),
                chunk_text=result["text"],
                page_number=result.get("page_number"),
                timestamp_start=result.get("metadata", {}).get("timestamp_start"),
                timestamp_end=result.get("metadata", {}).get("timestamp_end"),
                relevance_score=result["score"],
            )
            citations.append(citation)

        return citations

    def build_rag_prompt(
        self,
        query: str,
        citations: List[Citation],
        user_mastery: Optional[Dict[int, float]] = None,
    ) -> str:
        """
        Build RAG prompt with retrieved context

        Args:
            query: User query
            citations: Retrieved citations
            user_mastery: Optional user mastery levels

        Returns:
            Formatted prompt with context
        """
        # Build context from citations
        context_parts = []
        for i, citation in enumerate(citations, 1):
            context_part = f"[Source {i}] {citation.module_title}\n"

            if citation.page_number:
                context_part += f"(Page {citation.page_number})\n"
            if citation.timestamp_start:
                mins = int(citation.timestamp_start // 60)
                secs = int(citation.timestamp_start % 60)
                context_part += f"(Timestamp {mins}:{secs:02d})\n"

            context_part += f"{citation.chunk_text}\n"
            context_parts.append(context_part)

        context = "\n".join(context_parts)

        # Add mastery context if available
        mastery_context = ""
        if user_mastery:
            avg_mastery = sum(user_mastery.values()) / len(user_mastery) if user_mastery else 0
            if avg_mastery < 0.3:
                mastery_context = "\n\nNote: The user is a beginner. Provide simple explanations with examples."
            elif avg_mastery < 0.7:
                mastery_context = "\n\nNote: The user has intermediate knowledge. Balance detail with clarity."
            else:
                mastery_context = "\n\nNote: The user has advanced knowledge. You can use technical terminology."

        # Build final prompt
        prompt = f"""You are an AI tutor helping a student learn from course materials.

Context from course materials:
{context}
{mastery_context}

Student's question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. If you reference information, cite the source number (e.g., [Source 1])
3. If the context doesn't contain the answer, say so clearly
4. Be conversational and encouraging
5. Use examples from the course materials when possible
6. If referring to a video timestamp or page number, mention it clearly

Answer:"""

        return prompt

    async def chat(
        self,
        query: str,
        user_id: int,
        course_id: int,
        vector_store_service,
        session_id: Optional[str] = None,
        module_id: Optional[int] = None,
        user_mastery: Optional[Dict[int, float]] = None,
    ) -> ChatMessage:
        """
        Process a chat query with RAG

        Args:
            query: User query
            user_id: User ID
            course_id: Course ID
            vector_store_service: Vector store service
            session_id: Optional session ID for context
            module_id: Optional module filter
            user_mastery: Optional user mastery levels

        Returns:
            ChatMessage with response and citations
        """
        # Retrieve relevant context
        citations = self.retrieve_context(
            query, course_id, vector_store_service, module_id=module_id
        )

        # Build RAG prompt
        rag_prompt = self.build_rag_prompt(query, citations, user_mastery)

        # Get conversation history
        session_key = f"{user_id}_{session_id}" if session_id else f"{user_id}_default"
        history = self.conversation_history.get(session_key, [])

        # Build messages for OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful AI tutor."},
        ]

        # Add recent history (last 5 messages)
        for msg in history[-5:]:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current query with RAG prompt
        messages.append({"role": "user", "content": rag_prompt})

        # Call OpenAI
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )

            assistant_message = response.choices[0].message.content

            # Extract concept IDs from citations
            concept_ids = []
            for citation in citations:
                # TODO: Get concept IDs from knowledge graph based on chunk content
                pass

            # Create response message
            response_msg = ChatMessage(
                role="assistant",
                content=assistant_message,
                citations=citations,
                concept_ids=concept_ids,
            )

            # Update conversation history
            if session_key not in self.conversation_history:
                self.conversation_history[session_key] = []

            self.conversation_history[session_key].append(
                ChatMessage(role="user", content=query)
            )
            self.conversation_history[session_key].append(response_msg)

            # Keep history limited (last 20 messages)
            if len(self.conversation_history[session_key]) > 20:
                self.conversation_history[session_key] = self.conversation_history[
                    session_key
                ][-20:]

            return response_msg

        except Exception as e:
            # Fallback response
            return ChatMessage(
                role="assistant",
                content=f"I encountered an error processing your question. Please try again. Error: {str(e)}",
                citations=[],
            )

    def clear_history(self, user_id: int, session_id: Optional[str] = None):
        """Clear conversation history"""
        session_key = f"{user_id}_{session_id}" if session_id else f"{user_id}_default"
        if session_key in self.conversation_history:
            del self.conversation_history[session_key]

    def get_history(
        self, user_id: int, session_id: Optional[str] = None
    ) -> List[ChatMessage]:
        """Get conversation history"""
        session_key = f"{user_id}_{session_id}" if session_id else f"{user_id}_default"
        return self.conversation_history.get(session_key, [])
