"""
Podcast Generator - Text-to-Podcast Synthesis (NotebookLM Effect)

Research alignment:
- Podcastfy Architecture: Scriptwriter → Dramatization → TTS Synthesis
- Director-Actor Model: Separate content from performance
- Multi-speaker Diarization: Host + Guest personas
- Natural Conversation: Disfluencies, back-channeling, interruptions

Key Components:
1. ScriptwriterAgent: Creates engaging dialogue from educational content
2. DramatizationEngine: Adds natural speech patterns and cues
3. TTSSynthesizer: Converts script to audio with multiple voices
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpeakerRole(str, Enum):
    """Roles in the podcast conversation"""
    HOST = "host"           # Main explainer/summarizer
    GUEST = "guest"         # Questioner/clarifier
    EXPERT = "expert"       # Deep technical insights
    SKEPTIC = "skeptic"     # Challenges assumptions


class VoiceStyle(str, Enum):
    """Voice characteristics for TTS"""
    WARM_PROFESSIONAL = "warm_professional"
    ENTHUSIASTIC = "enthusiastic"
    THOUGHTFUL = "thoughtful"
    CURIOUS = "curious"
    AUTHORITATIVE = "authoritative"


@dataclass
class DialogueTurn:
    """A single turn in the podcast dialogue"""
    speaker: SpeakerRole
    text: str
    emotion: Optional[str] = None  # e.g., "curious", "excited", "thoughtful"
    stage_direction: Optional[str] = None  # e.g., "[laughs]", "[pauses]"
    emphasis_words: List[str] = field(default_factory=list)


@dataclass
class PodcastScript:
    """Complete podcast script with dialogue and metadata"""
    title: str
    topic: str
    introduction: str
    dialogue: List[DialogueTurn]
    conclusion: str
    duration_estimate_minutes: int
    key_concepts: List[str]
    target_audience: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PodcastEpisode:
    """Generated podcast episode with audio"""
    script: PodcastScript
    audio_url: Optional[str] = None
    audio_data: Optional[bytes] = None
    transcript_with_timestamps: Optional[List[Dict[str, Any]]] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)


class ScriptwriterAgent:
    """
    Creates engaging dialogue scripts from educational content

    The Scriptwriter separates content extraction from dramatization:
    1. Researcher role: Extracts key themes and concepts
    2. Director role: Structures the conversation flow
    3. Writer role: Creates natural dialogue
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.8)

    async def create_script(
        self,
        content: str,
        topic: str,
        duration_minutes: int = 10,
        style: str = "educational_casual",
        learner_state: Optional[Dict[str, Any]] = None
    ) -> PodcastScript:
        """
        Create a podcast script from educational content

        Args:
            content: Source educational content
            topic: Main topic
            duration_minutes: Target duration
            style: Conversation style
            learner_state: Optional state about what learner already knows

        Returns:
            PodcastScript with complete dialogue
        """
        # Calculate approximate word count (150 words/minute for natural speech)
        target_words = duration_minutes * 150

        # Build context about learner's state
        learner_context = ""
        if learner_state:
            understood = learner_state.get("understood_concepts", [])
            struggling = learner_state.get("struggling_with", [])
            if understood:
                learner_context += f"\nThe listener already understands: {', '.join(understood)}"
            if struggling:
                learner_context += f"\nThe listener is struggling with: {', '.join(struggling)}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert podcast scriptwriter for educational content.

Create engaging, natural dialogue between a Host and Guest that explains complex topics
in an accessible way. The conversation should feel like two curious friends discussing
something fascinating.

STYLE GUIDELINES:
- Host: Warm, knowledgeable, uses analogies and examples
- Guest: Curious, asks clarifying questions, represents the learner
- Include natural speech patterns: "So...", "Well...", "I mean..."
- Add moments of genuine interest: "Oh, that's interesting!"
- Use back-channeling: "Right", "Mm-hmm", "Sure"
- Include brief pauses for emphasis: [pause]
- Add occasional disfluencies for realism: "It's like... you know..."
- Make complex ideas concrete with everyday examples

STRUCTURE:
1. Hook (grab attention with surprising fact or question)
2. Foundation (establish base knowledge)
3. Deep Dive (explore main concepts)
4. Examples (concrete applications)
5. Synthesis (tie it together)
6. Takeaway (actionable insight)

OUTPUT FORMAT (JSON):
{{
    "title": "Episode title",
    "introduction": "Brief intro (Host only)",
    "dialogue": [
        {{"speaker": "host", "text": "...", "emotion": "enthusiastic", "stage_direction": null}},
        {{"speaker": "guest", "text": "...", "emotion": "curious", "stage_direction": "[laughs]"}}
    ],
    "conclusion": "Brief outro (Host only)",
    "key_concepts": ["list of concepts covered"],
    "target_audience": "description"
}}"""),
            ("human", """Create a {duration}-minute podcast script about:

TOPIC: {topic}

SOURCE CONTENT:
{content}
{learner_context}

Target approximately {word_count} words total.
Make it engaging, educational, and natural-sounding.

Generate the script as JSON:""")
        ])

        try:
            messages = prompt.format_messages(
                duration=duration_minutes,
                topic=topic,
                content=content[:4000],  # Limit context
                learner_context=learner_context,
                word_count=target_words
            )

            response = await self.llm.ainvoke(messages)

            # Parse JSON response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            script_data = json.loads(response_text.strip())

            # Convert to DialogueTurn objects
            dialogue = []
            for turn in script_data.get("dialogue", []):
                speaker = SpeakerRole.HOST if turn.get("speaker") == "host" else SpeakerRole.GUEST
                dialogue.append(DialogueTurn(
                    speaker=speaker,
                    text=turn.get("text", ""),
                    emotion=turn.get("emotion"),
                    stage_direction=turn.get("stage_direction")
                ))

            return PodcastScript(
                title=script_data.get("title", f"Episode: {topic}"),
                topic=topic,
                introduction=script_data.get("introduction", ""),
                dialogue=dialogue,
                conclusion=script_data.get("conclusion", ""),
                duration_estimate_minutes=duration_minutes,
                key_concepts=script_data.get("key_concepts", []),
                target_audience=script_data.get("target_audience", "general learners")
            )

        except Exception as e:
            logger.error(f"Error creating script: {e}")
            # Return minimal fallback script
            return PodcastScript(
                title=f"Episode: {topic}",
                topic=topic,
                introduction=f"Today we're discussing {topic}.",
                dialogue=[
                    DialogueTurn(speaker=SpeakerRole.HOST, text=content[:500])
                ],
                conclusion="Thanks for listening!",
                duration_estimate_minutes=duration_minutes,
                key_concepts=[topic],
                target_audience="general learners"
            )


class DramatizationEngine:
    """
    Enhances scripts with natural speech patterns

    Adds:
    - Disfluencies (um, uh, like)
    - Back-channeling (right, sure, mm-hmm)
    - Interruptions and overlaps
    - Emotional markers
    - Pause indicators
    """

    def __init__(self, intensity: float = 0.3):
        """
        Initialize dramatization engine

        Args:
            intensity: How much dramatization to add (0-1)
        """
        self.intensity = intensity

        # Natural speech additions
        self.disfluencies = ["um", "uh", "like", "you know", "I mean", "so"]
        self.back_channels = ["right", "sure", "mm-hmm", "yeah", "exactly", "interesting"]
        self.reactions = ["Oh!", "Huh.", "Wow.", "Really?", "That's fascinating."]
        self.thinking_markers = ["Well...", "Let me think...", "So basically...", "Hmm..."]

    def dramatize(self, script: PodcastScript) -> PodcastScript:
        """
        Add natural speech patterns to script

        Args:
            script: Original script

        Returns:
            Enhanced script with dramatization
        """
        import random
        random.seed(42)  # Reproducible results

        enhanced_dialogue = []

        for i, turn in enumerate(script.dialogue):
            enhanced_text = turn.text

            # Add occasional disfluency at start
            if random.random() < self.intensity and len(enhanced_text) > 50:
                disfluency = random.choice(self.thinking_markers)
                enhanced_text = f"{disfluency} {enhanced_text}"

            # Add stage directions for long responses
            if len(enhanced_text) > 200 and not turn.stage_direction:
                turn.stage_direction = random.choice(["[thoughtfully]", "[nodding]", "[leaning in]"])

            # Add back-channel response from other speaker occasionally
            if i > 0 and random.random() < self.intensity * 0.5:
                # Insert a brief acknowledgment
                prev_speaker = script.dialogue[i-1].speaker
                if prev_speaker != turn.speaker:
                    back_channel = random.choice(self.back_channels)
                    enhanced_dialogue.append(DialogueTurn(
                        speaker=turn.speaker,
                        text=back_channel,
                        emotion="engaged",
                        stage_direction="[brief]"
                    ))

            enhanced_dialogue.append(DialogueTurn(
                speaker=turn.speaker,
                text=enhanced_text,
                emotion=turn.emotion or self._infer_emotion(enhanced_text),
                stage_direction=turn.stage_direction
            ))

        script.dialogue = enhanced_dialogue
        return script

    def _infer_emotion(self, text: str) -> str:
        """Infer emotion from text content"""
        text_lower = text.lower()

        if "?" in text:
            return "curious"
        elif any(word in text_lower for word in ["amazing", "fascinating", "incredible", "wow"]):
            return "excited"
        elif any(word in text_lower for word in ["important", "key", "crucial", "remember"]):
            return "emphatic"
        elif any(word in text_lower for word in ["think", "consider", "perhaps", "maybe"]):
            return "thoughtful"
        else:
            return "conversational"


class TTSSynthesizer:
    """
    Converts podcast scripts to audio using TTS

    Supports multiple backends:
    - OpenAI TTS (default)
    - ElevenLabs (higher quality)
    - Edge TTS (free alternative)
    """

    def __init__(
        self,
        backend: str = "openai",
        openai_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None
    ):
        self.backend = backend
        self.openai_api_key = openai_api_key
        self.elevenlabs_api_key = elevenlabs_api_key

        # Voice mappings for different backends
        self.voice_map = {
            "openai": {
                SpeakerRole.HOST: "alloy",      # Warm, professional
                SpeakerRole.GUEST: "nova",      # Curious, friendly
                SpeakerRole.EXPERT: "onyx",     # Deep, authoritative
                SpeakerRole.SKEPTIC: "echo",    # Questioning
            },
            "elevenlabs": {
                SpeakerRole.HOST: "Rachel",
                SpeakerRole.GUEST: "Josh",
                SpeakerRole.EXPERT: "Adam",
                SpeakerRole.SKEPTIC: "Bella",
            }
        }

    async def synthesize(
        self,
        script: PodcastScript,
        output_format: str = "mp3"
    ) -> Optional[bytes]:
        """
        Synthesize audio from podcast script

        Args:
            script: Podcast script to synthesize
            output_format: Audio format (mp3, wav)

        Returns:
            Audio data as bytes, or None if synthesis fails
        """
        if self.backend == "openai":
            return await self._synthesize_openai(script)
        elif self.backend == "elevenlabs":
            return await self._synthesize_elevenlabs(script)
        elif self.backend == "edge":
            return await self._synthesize_edge(script)
        else:
            logger.error(f"Unknown TTS backend: {self.backend}")
            return None

    async def _synthesize_openai(self, script: PodcastScript) -> Optional[bytes]:
        """Synthesize using OpenAI TTS"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            audio_segments = []

            # Synthesize introduction
            if script.introduction:
                response = await client.audio.speech.create(
                    model="tts-1",
                    voice=self.voice_map["openai"][SpeakerRole.HOST],
                    input=script.introduction
                )
                audio_segments.append(response.content)

            # Synthesize dialogue
            for turn in script.dialogue:
                voice = self.voice_map["openai"].get(turn.speaker, "alloy")

                # Add stage direction as SSML-like pause if present
                text = turn.text
                if turn.stage_direction and "[pause]" in turn.stage_direction:
                    text = text + "..."  # Simple pause simulation

                response = await client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text
                )
                audio_segments.append(response.content)

            # Synthesize conclusion
            if script.conclusion:
                response = await client.audio.speech.create(
                    model="tts-1",
                    voice=self.voice_map["openai"][SpeakerRole.HOST],
                    input=script.conclusion
                )
                audio_segments.append(response.content)

            # Combine audio segments
            # Note: For proper audio concatenation, use pydub or similar
            combined = b"".join(audio_segments)
            return combined

        except ImportError:
            logger.warning("OpenAI client not available for TTS")
            return None
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis error: {e}")
            return None

    async def _synthesize_elevenlabs(self, script: PodcastScript) -> Optional[bytes]:
        """Synthesize using ElevenLabs API"""
        if not self.elevenlabs_api_key:
            logger.warning("ElevenLabs API key not provided")
            return None

        try:
            import httpx

            audio_segments = []
            base_url = "https://api.elevenlabs.io/v1/text-to-speech"

            async with httpx.AsyncClient() as client:
                for turn in script.dialogue:
                    voice_id = self._get_elevenlabs_voice_id(turn.speaker)

                    response = await client.post(
                        f"{base_url}/{voice_id}",
                        headers={
                            "xi-api-key": self.elevenlabs_api_key,
                            "Content-Type": "application/json"
                        },
                        json={
                            "text": turn.text,
                            "model_id": "eleven_monolingual_v1",
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.75
                            }
                        },
                        timeout=60.0
                    )

                    if response.status_code == 200:
                        audio_segments.append(response.content)

            return b"".join(audio_segments) if audio_segments else None

        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            return None

    async def _synthesize_edge(self, script: PodcastScript) -> Optional[bytes]:
        """Synthesize using Edge TTS (free)"""
        try:
            import edge_tts

            audio_segments = []

            voice_map = {
                SpeakerRole.HOST: "en-US-GuyNeural",
                SpeakerRole.GUEST: "en-US-JennyNeural",
                SpeakerRole.EXPERT: "en-GB-RyanNeural",
                SpeakerRole.SKEPTIC: "en-US-AriaNeural",
            }

            for turn in script.dialogue:
                voice = voice_map.get(turn.speaker, "en-US-GuyNeural")
                communicate = edge_tts.Communicate(turn.text, voice)

                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]

                audio_segments.append(audio_data)

            return b"".join(audio_segments) if audio_segments else None

        except ImportError:
            logger.warning("edge-tts not installed")
            return None
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return None

    def _get_elevenlabs_voice_id(self, speaker: SpeakerRole) -> str:
        """Get ElevenLabs voice ID for speaker"""
        # Default voice IDs (these are examples, use actual IDs from ElevenLabs)
        voice_ids = {
            SpeakerRole.HOST: "21m00Tcm4TlvDq8ikWAM",     # Rachel
            SpeakerRole.GUEST: "TxGEqnHWrfWFTfGW9XjX",    # Josh
            SpeakerRole.EXPERT: "pNInz6obpgDQGcFmaJgB",   # Adam
            SpeakerRole.SKEPTIC: "EXAVITQu4vr4xnSDxMaL",  # Bella
        }
        return voice_ids.get(speaker, voice_ids[SpeakerRole.HOST])


class PodcastGenerator:
    """
    Complete podcast generation pipeline

    Orchestrates:
    1. Script creation from content
    2. Dramatization for natural feel
    3. TTS synthesis for audio
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tts_backend: str = "openai",
        dramatization_intensity: float = 0.3
    ):
        self.scriptwriter = ScriptwriterAgent(llm)
        self.dramatizer = DramatizationEngine(dramatization_intensity)
        self.synthesizer = TTSSynthesizer(backend=tts_backend)

    async def generate(
        self,
        content: str,
        topic: str,
        duration_minutes: int = 10,
        style: str = "educational_casual",
        learner_state: Optional[Dict[str, Any]] = None,
        synthesize_audio: bool = True
    ) -> PodcastEpisode:
        """
        Generate a complete podcast episode

        Args:
            content: Source educational content
            topic: Main topic
            duration_minutes: Target duration
            style: Conversation style
            learner_state: Optional learner context
            synthesize_audio: Whether to generate audio

        Returns:
            PodcastEpisode with script and optional audio
        """
        logger.info(f"Generating {duration_minutes}-minute podcast for: {topic}")

        # Step 1: Create script
        logger.debug("Creating script...")
        script = await self.scriptwriter.create_script(
            content=content,
            topic=topic,
            duration_minutes=duration_minutes,
            style=style,
            learner_state=learner_state
        )

        # Step 2: Dramatize
        logger.debug("Adding dramatization...")
        script = self.dramatizer.dramatize(script)

        # Step 3: Synthesize audio (optional)
        audio_data = None
        if synthesize_audio:
            logger.debug("Synthesizing audio...")
            audio_data = await self.synthesizer.synthesize(script)

        return PodcastEpisode(
            script=script,
            audio_data=audio_data,
            generated_at=datetime.utcnow()
        )

    def get_script_text(self, script: PodcastScript) -> str:
        """Get plain text version of script"""
        lines = [f"# {script.title}\n"]
        lines.append(f"**Introduction:** {script.introduction}\n")

        for turn in script.dialogue:
            speaker = "HOST" if turn.speaker == SpeakerRole.HOST else "GUEST"
            direction = f" {turn.stage_direction}" if turn.stage_direction else ""
            lines.append(f"**{speaker}:**{direction} {turn.text}\n")

        lines.append(f"**Conclusion:** {script.conclusion}")
        return "\n".join(lines)


# Global instance (lazy initialization)
_podcast_generator: Optional[PodcastGenerator] = None


def get_podcast_generator_instance() -> PodcastGenerator:
    """Get or create the podcast generator instance (lazy)"""
    global _podcast_generator
    if _podcast_generator is None:
        _podcast_generator = PodcastGenerator()
    return _podcast_generator


async def get_podcast_generator() -> PodcastGenerator:
    """Dependency injection"""
    return get_podcast_generator_instance()
