from typing import List, Any, Optional, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class MockChatModel(BaseChatModel):
    """
    A mock chat model that returns deterministic responses.
    Useful for offline testing and simulation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a mock response based on the last message content.
        """
        last_message = messages[-1].content.lower()
        
        # Simple heuristic response logic
        if "socratic" in last_message or "tutor" in str(messages[0].content).lower():
            response_content = "That is an interesting perspective. Can you explain why you think that is the case? (Mock Socratic Response)"
        elif "explain" in last_message:
            response_content = "Here is a simplified explanation: Concept X relates to Concept Y... (Mock Explanation)"
        elif "json" in last_message or "structured" in last_message:
            response_content = '{"key": "value", "reasoning": "mock reasoning"}'
        else:
            response_content = "I am a mock AI. receiving transmission."

        generation = ChatGeneration(message=AIMessage(content=response_content))
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": "mock-gpt-4o"}

class MockEmbeddings:
    """
    Deterministic mock embeddings for offline vector search testing.
    """
    def __init__(self, size: int = 1536):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        # Deterministic hashing to generate a pseudo-random vector from text
        # This ensures the same text always gets the same vector
        import hashlib
        import random
        
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        random.seed(seed)
        
        # Generate normalized vector
        vector = [random.uniform(-1.0, 1.0) for _ in range(self.size)]
        
        # Normalize
        norm = sum(x*x for x in vector) ** 0.5
        return [x/norm for x in vector]
