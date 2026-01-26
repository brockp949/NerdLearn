from typing import Optional
import os
from langchain_core.language_models.chat_models import BaseChatModel

# Safe imports to handle missing packages gracefully if not fully installed yet
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

class LLMProvider:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class LLMFactory:
    """
    Abstracts the creation of Language Model clients for the Swarm.
    Supports OpenAI, Anthropic (Claude), and Google (Gemini).
    """
    
    @staticmethod
    def create_llm(provider: str, model: str, temperature: float = 0.7, api_key: Optional[str] = None) -> BaseChatModel:
        """
        Creates and returns a configured LangChain Chat Model.
        """
        if provider == LLMProvider.OPENAI:
            if not ChatOpenAI:
                raise ImportError("langchain-openai is not installed.")
            return ChatOpenAI(
                model=model, 
                temperature=temperature,
                api_key=api_key or os.environ.get("OPENAI_API_KEY")
            )
        
        elif provider == LLMProvider.ANTHROPIC:
            if not ChatAnthropic:
                raise ImportError("langchain-anthropic is not installed.")
            return ChatAnthropic(
                model=model, 
                temperature=temperature,
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
            
        elif provider == LLMProvider.GEMINI:
            if not ChatGoogleGenerativeAI:
                raise ImportError("langchain-google-genai is not installed.")
            # Google often requires explicit conversion adjustment for some models
            return ChatGoogleGenerativeAI(
                model=model, 
                temperature=temperature,
                google_api_key=api_key or os.environ.get("GOOGLE_API_KEY"),
                convert_system_message_to_human=True 
            )
            
        raise ValueError(f"Unknown LLM Provider: {provider}")
