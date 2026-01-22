"""
Audio Overview Generation Service

Generates audio summaries and overviews of course content using
text-to-speech services (ElevenLabs, OpenAI TTS).

Features:
- Course/module overview generation
- Multiple voice options
- Audio caching
- Chunked generation for long content
- Background music/ambient mixing (optional)
"""

import os
import io
import hashlib
import logging
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import asyncio
import aiohttp
import json
from pathlib import Path

from ..config import config

logger = logging.getLogger(__name__)


class TTSProvider(str, Enum):
    """Available TTS providers"""
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    LOCAL = "local"  # For testing/fallback


class VoiceStyle(str, Enum):
    """Voice styles for different content types"""
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"


@dataclass
class VoiceConfig:
    """Configuration for TTS voice"""
    provider: TTSProvider
    voice_id: str
    style: VoiceStyle = VoiceStyle.EDUCATIONAL
    speed: float = 1.0
    pitch: float = 1.0
    stability: float = 0.5  # ElevenLabs specific
    similarity_boost: float = 0.75  # ElevenLabs specific


@dataclass
class AudioResult:
    """Result of audio generation"""
    audio_data: bytes
    duration_seconds: float
    format: str
    word_count: int
    cache_key: Optional[str] = None


class AudioService:
    """
    Main audio generation service.

    Supports multiple TTS providers with automatic fallback.
    """

    # Default voice configurations
    DEFAULT_VOICES = {
        TTSProvider.ELEVENLABS: {
            VoiceStyle.PROFESSIONAL: "21m00Tcm4TlvDq8ikWAM",  # Rachel
            VoiceStyle.CONVERSATIONAL: "EXAVITQu4vr4xnSDxMaL",  # Bella
            VoiceStyle.EDUCATIONAL: "pNInz6obpgDQGcFmaJgB",  # Adam
            VoiceStyle.ENTHUSIASTIC: "jsCqWAovK2LkecY7zXl4",  # Nicole
            VoiceStyle.CALM: "MF3mGyEYCl7XYWbV9V6O",  # Elli
        },
        TTSProvider.OPENAI: {
            VoiceStyle.PROFESSIONAL: "alloy",
            VoiceStyle.CONVERSATIONAL: "nova",
            VoiceStyle.EDUCATIONAL: "onyx",
            VoiceStyle.ENTHUSIASTIC: "shimmer",
            VoiceStyle.CALM: "echo",
        }
    }

    # Rate limits (requests per minute)
    RATE_LIMITS = {
        TTSProvider.ELEVENLABS: 10,
        TTSProvider.OPENAI: 50,
        TTSProvider.LOCAL: 1000,
    }

    # Character limits per request
    CHAR_LIMITS = {
        TTSProvider.ELEVENLABS: 5000,
        TTSProvider.OPENAI: 4096,
        TTSProvider.LOCAL: 10000,
    }

    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        preferred_provider: TTSProvider = TTSProvider.ELEVENLABS
    ):
        """
        Initialize audio service.

        Args:
            elevenlabs_api_key: ElevenLabs API key
            openai_api_key: OpenAI API key
            cache_dir: Directory for audio caching
            preferred_provider: Preferred TTS provider
        """
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.preferred_provider = preferred_provider
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/nerdlearn_audio_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Request tracking for rate limiting
        self._request_times: Dict[TTSProvider, List[float]] = {
            provider: [] for provider in TTSProvider
        }

    def _get_available_providers(self) -> List[TTSProvider]:
        """Get list of available providers with API keys configured"""
        providers = []
        if self.elevenlabs_api_key:
            providers.append(TTSProvider.ELEVENLABS)
        if self.openai_api_key:
            providers.append(TTSProvider.OPENAI)
        providers.append(TTSProvider.LOCAL)  # Always available
        return providers

    def _get_cache_key(self, text: str, voice_config: VoiceConfig) -> str:
        """Generate cache key for audio"""
        content = f"{text}:{voice_config.provider}:{voice_config.voice_id}:{voice_config.speed}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Get cached audio if available"""
        cache_path = self.cache_dir / f"{cache_key}.mp3"
        if cache_path.exists():
            return cache_path.read_bytes()
        return None

    def _cache_audio(self, cache_key: str, audio_data: bytes):
        """Cache audio data"""
        cache_path = self.cache_dir / f"{cache_key}.mp3"
        cache_path.write_bytes(audio_data)

    async def _check_rate_limit(self, provider: TTSProvider) -> bool:
        """Check if we're within rate limits"""
        import time
        current_time = time.time()
        minute_ago = current_time - 60

        # Clean old requests
        self._request_times[provider] = [
            t for t in self._request_times[provider] if t > minute_ago
        ]

        return len(self._request_times[provider]) < self.RATE_LIMITS[provider]

    async def _record_request(self, provider: TTSProvider):
        """Record a request for rate limiting"""
        import time
        self._request_times[provider].append(time.time())

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks respecting sentence boundaries"""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by sentences
        sentences = text.replace(".", ".|").replace("!", "!|").replace("?", "?|").split("|")

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def generate_audio(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        use_cache: bool = True
    ) -> AudioResult:
        """
        Generate audio from text.

        Args:
            text: Text to convert to speech
            voice_config: Voice configuration (uses defaults if not provided)
            use_cache: Whether to use/store cached audio

        Returns:
            AudioResult with audio data and metadata
        """
        if not voice_config:
            voice_config = VoiceConfig(
                provider=self.preferred_provider,
                voice_id=self.DEFAULT_VOICES.get(
                    self.preferred_provider, {}
                ).get(VoiceStyle.EDUCATIONAL, "default"),
                style=VoiceStyle.EDUCATIONAL
            )

        # Check cache
        cache_key = self._get_cache_key(text, voice_config)
        if use_cache:
            cached = self._get_cached_audio(cache_key)
            if cached:
                logger.info(f"Using cached audio for key: {cache_key}")
                return AudioResult(
                    audio_data=cached,
                    duration_seconds=self._estimate_duration(text),
                    format="mp3",
                    word_count=len(text.split()),
                    cache_key=cache_key
                )

        # Get available providers
        available = self._get_available_providers()
        if voice_config.provider not in available:
            # Fallback to first available
            voice_config.provider = available[0]
            voice_config.voice_id = self.DEFAULT_VOICES.get(
                voice_config.provider, {}
            ).get(voice_config.style, "default")

        # Check rate limit
        if not await self._check_rate_limit(voice_config.provider):
            raise Exception(f"Rate limit exceeded for {voice_config.provider}")

        # Chunk text if needed
        max_chars = self.CHAR_LIMITS[voice_config.provider]
        chunks = self._chunk_text(text, max_chars)

        # Generate audio for each chunk
        audio_chunks = []
        for chunk in chunks:
            audio_data = await self._generate_chunk(chunk, voice_config)
            audio_chunks.append(audio_data)
            await self._record_request(voice_config.provider)

        # Combine chunks
        if len(audio_chunks) == 1:
            combined_audio = audio_chunks[0]
        else:
            combined_audio = self._combine_audio_chunks(audio_chunks)

        # Cache result
        if use_cache:
            self._cache_audio(cache_key, combined_audio)

        return AudioResult(
            audio_data=combined_audio,
            duration_seconds=self._estimate_duration(text),
            format="mp3",
            word_count=len(text.split()),
            cache_key=cache_key
        )

    async def _generate_chunk(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> bytes:
        """Generate audio for a single chunk"""
        if voice_config.provider == TTSProvider.ELEVENLABS:
            return await self._generate_elevenlabs(text, voice_config)
        elif voice_config.provider == TTSProvider.OPENAI:
            return await self._generate_openai(text, voice_config)
        else:
            return await self._generate_local(text, voice_config)

    async def _generate_elevenlabs(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> bytes:
        """Generate audio using ElevenLabs API"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config.voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_api_key
        }

        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": voice_config.stability,
                "similarity_boost": voice_config.similarity_boost,
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ElevenLabs API error: {error_text}")
                    raise Exception(f"ElevenLabs API error: {response.status}")
                return await response.read()

    async def _generate_openai(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> bytes:
        """Generate audio using OpenAI TTS API"""
        url = "https://api.openai.com/v1/audio/speech"

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice_config.voice_id,
            "response_format": "mp3",
            "speed": voice_config.speed
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI TTS API error: {error_text}")
                    raise Exception(f"OpenAI TTS API error: {response.status}")
                return await response.read()

    async def _generate_local(
        self,
        text: str,
        voice_config: VoiceConfig
    ) -> bytes:
        """Generate placeholder audio for local/testing"""
        # Return a simple placeholder
        # In production, could use a local TTS engine like pyttsx3
        logger.warning("Using local placeholder audio generation")

        # Generate a simple sine wave as placeholder
        import struct
        import math

        sample_rate = 22050
        duration = len(text.split()) * 0.4  # ~0.4 seconds per word
        num_samples = int(sample_rate * duration)

        # Generate simple tone
        audio_data = []
        for i in range(num_samples):
            sample = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sample_rate))
            audio_data.append(struct.pack('<h', sample))

        return b''.join(audio_data)

    def _combine_audio_chunks(self, chunks: List[bytes]) -> bytes:
        """Combine multiple audio chunks into one"""
        # Simple concatenation for MP3
        # In production, use pydub for proper audio processing
        return b''.join(chunks)

    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration from text"""
        words = len(text.split())
        words_per_minute = 150  # Average speaking rate
        return (words / words_per_minute) * 60


class AudioOverviewGenerator:
    """
    Generates audio overviews for courses and modules.

    Creates engaging audio summaries suitable for:
    - Podcast-style course introductions
    - Module summaries
    - Concept explanations
    - Quick review audio
    """

    def __init__(self, audio_service: Optional[AudioService] = None):
        """Initialize the overview generator"""
        self.audio_service = audio_service or AudioService()

    async def generate_course_overview(
        self,
        course_title: str,
        course_description: str,
        modules: List[Dict[str, str]],
        key_concepts: List[str],
        style: VoiceStyle = VoiceStyle.EDUCATIONAL
    ) -> AudioResult:
        """
        Generate an audio overview for a course.

        Args:
            course_title: Course title
            course_description: Course description
            modules: List of {"title": str, "summary": str}
            key_concepts: List of key concept names
            style: Voice style to use

        Returns:
            AudioResult with the generated overview
        """
        # Create overview script
        script = self._create_course_overview_script(
            course_title, course_description, modules, key_concepts
        )

        # Generate audio
        voice_config = VoiceConfig(
            provider=self.audio_service.preferred_provider,
            voice_id=self.audio_service.DEFAULT_VOICES.get(
                self.audio_service.preferred_provider, {}
            ).get(style, "default"),
            style=style
        )

        return await self.audio_service.generate_audio(script, voice_config)

    async def generate_module_summary(
        self,
        module_title: str,
        content_summary: str,
        key_points: List[str],
        concepts_covered: List[str],
        style: VoiceStyle = VoiceStyle.EDUCATIONAL
    ) -> AudioResult:
        """
        Generate an audio summary for a module.

        Args:
            module_title: Module title
            content_summary: Summary of module content
            key_points: Key takeaways
            concepts_covered: Concepts covered in this module
            style: Voice style

        Returns:
            AudioResult with the generated summary
        """
        script = self._create_module_summary_script(
            module_title, content_summary, key_points, concepts_covered
        )

        voice_config = VoiceConfig(
            provider=self.audio_service.preferred_provider,
            voice_id=self.audio_service.DEFAULT_VOICES.get(
                self.audio_service.preferred_provider, {}
            ).get(style, "default"),
            style=style
        )

        return await self.audio_service.generate_audio(script, voice_config)

    async def generate_concept_explanation(
        self,
        concept_name: str,
        definition: str,
        examples: List[str],
        related_concepts: List[str],
        style: VoiceStyle = VoiceStyle.CONVERSATIONAL
    ) -> AudioResult:
        """
        Generate an audio explanation for a concept.

        Args:
            concept_name: Name of the concept
            definition: Definition/explanation
            examples: Example applications
            related_concepts: Related concept names
            style: Voice style

        Returns:
            AudioResult with the concept explanation
        """
        script = self._create_concept_explanation_script(
            concept_name, definition, examples, related_concepts
        )

        voice_config = VoiceConfig(
            provider=self.audio_service.preferred_provider,
            voice_id=self.audio_service.DEFAULT_VOICES.get(
                self.audio_service.preferred_provider, {}
            ).get(style, "default"),
            style=style
        )

        return await self.audio_service.generate_audio(script, voice_config)

    async def generate_quick_review(
        self,
        topic: str,
        bullet_points: List[str],
        quiz_questions: Optional[List[Dict[str, str]]] = None,
        style: VoiceStyle = VoiceStyle.ENTHUSIASTIC
    ) -> AudioResult:
        """
        Generate a quick review audio.

        Args:
            topic: Topic being reviewed
            bullet_points: Key points to review
            quiz_questions: Optional quiz questions with answers
            style: Voice style

        Returns:
            AudioResult with the quick review
        """
        script = self._create_quick_review_script(topic, bullet_points, quiz_questions)

        voice_config = VoiceConfig(
            provider=self.audio_service.preferred_provider,
            voice_id=self.audio_service.DEFAULT_VOICES.get(
                self.audio_service.preferred_provider, {}
            ).get(style, "default"),
            style=style
        )

        return await self.audio_service.generate_audio(script, voice_config)

    def _create_course_overview_script(
        self,
        title: str,
        description: str,
        modules: List[Dict[str, str]],
        concepts: List[str]
    ) -> str:
        """Create script for course overview"""
        script_parts = [
            f"Welcome to {title}.",
            "",
            description,
            "",
            f"In this course, you'll explore {len(modules)} modules covering essential topics.",
        ]

        # Add module summaries
        for i, module in enumerate(modules[:5], 1):  # Limit to 5 for brevity
            script_parts.append(f"Module {i}: {module.get('title', 'Untitled')}.")
            if module.get('summary'):
                script_parts.append(module['summary'])

        # Add key concepts
        if concepts:
            script_parts.append("")
            script_parts.append(f"Key concepts you'll master include: {', '.join(concepts[:7])}.")

        script_parts.append("")
        script_parts.append("Let's begin your learning journey!")

        return " ".join(script_parts)

    def _create_module_summary_script(
        self,
        title: str,
        summary: str,
        key_points: List[str],
        concepts: List[str]
    ) -> str:
        """Create script for module summary"""
        script_parts = [
            f"Module Summary: {title}.",
            "",
            summary,
            "",
        ]

        if key_points:
            script_parts.append("Key takeaways from this module:")
            for i, point in enumerate(key_points[:5], 1):
                script_parts.append(f"Point {i}: {point}")

        if concepts:
            script_parts.append("")
            script_parts.append(f"Concepts covered: {', '.join(concepts[:5])}.")

        script_parts.append("")
        script_parts.append("Great progress! Keep up the learning momentum.")

        return " ".join(script_parts)

    def _create_concept_explanation_script(
        self,
        name: str,
        definition: str,
        examples: List[str],
        related: List[str]
    ) -> str:
        """Create script for concept explanation"""
        script_parts = [
            f"Let's explore the concept of {name}.",
            "",
            definition,
        ]

        if examples:
            script_parts.append("")
            script_parts.append("Here are some examples:")
            for example in examples[:3]:
                script_parts.append(example)

        if related:
            script_parts.append("")
            script_parts.append(f"This concept relates to: {', '.join(related[:4])}.")

        return " ".join(script_parts)

    def _create_quick_review_script(
        self,
        topic: str,
        points: List[str],
        questions: Optional[List[Dict[str, str]]]
    ) -> str:
        """Create script for quick review"""
        script_parts = [
            f"Quick review time! Let's refresh your knowledge of {topic}.",
            "",
            "Remember these key points:",
        ]

        for point in points[:5]:
            script_parts.append(point)

        if questions:
            script_parts.append("")
            script_parts.append("Let's test your understanding with a quick quiz.")
            for q in questions[:3]:
                script_parts.append(f"Question: {q.get('question', '')}")
                script_parts.append("Think about your answer.")
                script_parts.append(f"The answer is: {q.get('answer', '')}")

        script_parts.append("")
        script_parts.append("Great job reviewing! You're making excellent progress.")

        return " ".join(script_parts)


# Singleton service instance
audio_service = AudioService()
audio_overview_generator = AudioOverviewGenerator(audio_service)
