"""
DSPy Configuration Module

Sets up the DSPy language model client and optimization settings.
Research alignment:
- Programmatic Prompt Optimization
- Metric-driven content transformation
"""
import dspy
import os
import logging
from typing import Optional, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global DSPy settings
_dspy_configured = False


def configure_dspy(
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    api_key: Optional[str] = None
):
    """
    Configure the default DSPy language model
    
    Args:
        model_name: The OpenAI model to use
        temperature: Sampling temperature
        api_key: OpenAI API key (defaults to settings or env)
    """
    global _dspy_configured
    
    try:
        api_key = api_key or settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            logger.warning("No OpenAI API key found for DSPy configuration")
            return

        # Configure OpenAI LM
        lm = dspy.OpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=4000
        )
        
        # Set as default
        dspy.settings.configure(lm=lm)
        _dspy_configured = True
        logger.info(f"DSPy configured with model: {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}", exc_info=True)
        raise


def get_dspy_lm():
    """Get the configured DSPy LM instance"""
    if not _dspy_configured:
        configure_dspy()
    return dspy.settings.lm
