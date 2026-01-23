"""
Podcast Service - Educational Audio Synthesis

Research alignment:
- "NotebookLM Effect": Conversational learning
- Multi-speaker synthesis: Host/Guest dynamic
- Emotional TTS: Using ElevenLabs/OpenAI via Podcastfy
"""
import logging
import json
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

from app.core.config import settings

# Try importing podcastfy
try:
    from podcastfy.client import generate_podcast
    HAS_PODCASTFY = True
except ImportError:
    HAS_PODCASTFY = False

logger = logging.getLogger(__name__)

class PodcastScript(BaseModel):
    title: str
    dialogue: List[Dict[str, str]]

class PodcastService:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        # Ensure env vars are set for podcastfy
        os.environ["OPENAI_API_KEY"] = self.api_key
        if settings.ELEVENLABS_API_KEY:
             os.environ["ELEVENLABS_API_KEY"] = settings.ELEVENLABS_API_KEY

    async def generate_podcast(self, content: str, topic: str = "Educational Topic") -> Dict[str, Any]:
        """
        Generate an audio podcast from text content using Podcastfy.
        """
        if not HAS_PODCASTFY:
            logger.warning("Podcastfy not installed. Returning mock response.")
            return self._get_mock_response(topic)

        try:
            # Podcastfy handles the pipeline: Text -> Script -> Audio
            # We run it in a thread/process executor since it might be blocking
            logger.info(f"Generating podcast for topic: {topic}")
            
            # Note: generate_podcast returns the path to the generated audio file
            # In a real async web app, running this purely blocking is bad, so we'd offload it.
            # but for now we wrap it.
            
            # Assuming podcastfy config allows passing text directly or we pass a file.
            # The library typically takes URLs or text.
            
            # Create a temp file for content if needed, or pass list of texts.
            # Based on library usage: generate_podcast(urls=[], text=content, ...)
            
            result_path = await asyncio.to_thread(
                generate_podcast,
                text=content,
                topic=topic,
                tts_model="openai" # or "elevenlabs" if configured
            )
            
            # For now, we assume result_path is the local file path.
            # In production, we'd upload this to S3 and return a URL.
            # Here specific logic would be needed to serve it.
            
            # Mocking the script return for now as podcastfy might not return the raw script object easily 
            # without parsing the transcript file it generates.
            
            return {
                "title": f"Podcast: {topic}",
                "script": [], # TODO: Parse transcript if available
                "audio_url": f"/static/audio/{os.path.basename(result_path)}", # Placeholder for serving
                "duration": 300 # Mock duration
            }

        except Exception as e:
            logger.error(f"Podcastfy generation failed: {e}", exc_info=True)
            return self._get_mock_response(topic, error=str(e))

    def _get_mock_response(self, topic: str, error: str = None) -> Dict[str, Any]:
        return {
            "title": f"Podcast: {topic} (Mock)",
            "script": [
                {"speaker": "System", "text": "Podcast generation unavailable."}
            ],
            "audio_url": "",
            "duration": 0,
            "error": error
        }
