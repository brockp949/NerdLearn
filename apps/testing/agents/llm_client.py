"""
Multi-Provider LLM Client for Agentic Testing

Supports both Claude (Anthropic) and Gemini (Google) models.
Provides a unified interface for all testing agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import os
import logging
import json

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    GEMINI = "gemini"
    MOCK = "mock"  # For testing


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    provider: LLMProvider
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load configuration from environment variables"""
        provider_str = os.getenv("AGENTIC_LLM_PROVIDER", "gemini").lower()
        provider = LLMProvider(provider_str)
        
        if provider == LLMProvider.CLAUDE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        elif provider == LLMProvider.GEMINI:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            api_key = None
            model = "mock"
        
        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096"))
        )


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a JSON response from the LLM"""
        pass


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude client"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Claude"""
        client = self._get_client()
        
        message = client.messages.create(
            model=self.config.model or "claude-sonnet-4-20250514",
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response using Claude"""
        json_prompt = prompt + "\n\nRespond with valid JSON only, no markdown formatting."
        response = await self.generate(json_prompt, **kwargs)
        
        # Clean up response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        return json.loads(response.strip())


class GeminiClient(BaseLLMClient):
    """Google Gemini client"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                self._client = genai.GenerativeModel(
                    self.config.model or "gemini-2.5-flash"
                )
            except ImportError:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        return self._client
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini"""
        model = self._get_client()
        
        generation_config = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response using Gemini"""
        json_prompt = prompt + "\n\nRespond with valid JSON only, no markdown formatting."
        response = await self.generate(json_prompt, **kwargs)
        
        # Clean up response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        return json.loads(response.strip())


class MockLLMClient(BaseLLMClient):
    """Mock client for testing without API calls"""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig(provider=LLMProvider.MOCK)
        self.call_count = 0
        self.last_prompt = None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return mock response"""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Return contextual mock responses
        if "verify" in prompt.lower() or "audit" in prompt.lower():
            return json.dumps({
                "passed": True,
                "confidence": 0.85,
                "reasoning": "Mock verification passed",
                "violations": []
            })
        elif "test" in prompt.lower() or "generate" in prompt.lower():
            return json.dumps([{
                "name": "mock_test",
                "description": "Mock generated test",
                "test_type": "unit",
                "coverage_score": 0.5,
                "triviality_score": 0.1
            }])
        else:
            return '{"status": "ok", "message": "Mock response"}'
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Return mock JSON response"""
        response = await self.generate(prompt, **kwargs)
        return json.loads(response)


def create_llm_client(config: Optional[LLMConfig] = None) -> BaseLLMClient:
    """
    Factory function to create the appropriate LLM client.
    
    Args:
        config: Optional configuration. If None, loads from environment.
    
    Returns:
        Configured LLM client
    """
    if config is None:
        config = LLMConfig.from_env()
    
    logger.info(f"Creating LLM client: provider={config.provider.value}, model={config.model}")
    
    if config.provider == LLMProvider.CLAUDE:
        return ClaudeClient(config)
    elif config.provider == LLMProvider.GEMINI:
        return GeminiClient(config)
    else:
        return MockLLMClient(config)


# Convenience function for quick access
def get_default_client() -> BaseLLMClient:
    """Get the default LLM client based on environment configuration"""
    return create_llm_client()
