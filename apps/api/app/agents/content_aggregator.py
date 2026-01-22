"""
Content Aggregation Pipeline - Autonomous Content Sourcing

Research alignment:
- YouTube Transcript Processing: yt-dlp for video/audio extraction
- Transcript Quality Filter: LLM-based scoring for pedagogical value
- Gap Filling: Source content to match generated syllabus
- Prerequisite Discovery: Extract prerequisite relationships from text

Key Components:
1. YouTubeTranscriptAgent: Extract and process video transcripts
2. TranscriptQualityFilter: Score content for educational value
3. ContentMatcher: Match content to Learning Outcomes
4. PrerequisiteExtractor: Discover concept dependencies
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import re
import logging
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Types of content that can be aggregated"""
    VIDEO = "video"
    ARTICLE = "article"
    TEXTBOOK = "textbook"
    DOCUMENTATION = "documentation"
    PODCAST = "podcast"
    CODE_EXAMPLE = "code_example"


@dataclass
class ContentQualityScore:
    """Quality assessment for a piece of content"""
    overall_score: float  # 0-1
    information_density: float
    coherence: float
    pedagogical_value: float
    technical_accuracy: float
    engagement_level: float
    is_suitable: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AggregatedContent:
    """Represents a piece of aggregated content"""
    id: str
    source_url: str
    content_type: ContentType
    title: str
    raw_text: str
    processed_text: Optional[str] = None
    quality_score: Optional[ContentQualityScore] = None
    concepts_covered: List[str] = field(default_factory=list)
    prerequisites_mentioned: List[str] = field(default_factory=list)
    duration_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    aggregated_at: datetime = field(default_factory=datetime.utcnow)


class YouTubeTranscriptRequest(BaseModel):
    """Request to fetch YouTube transcript"""
    video_url: str = Field(..., description="YouTube video URL")
    language: str = Field(default="en", description="Preferred language")
    include_auto_captions: bool = Field(default=True, description="Include auto-generated captions")


class YouTubeTranscriptAgent:
    """
    Agent for extracting and processing YouTube transcripts

    Uses youtube_transcript_api for transcript extraction.
    In production, can be extended with yt-dlp for more robust extraction.
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    async def fetch_transcript(
        self,
        video_url: str,
        language: str = "en",
        include_auto_captions: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch transcript from YouTube video

        Args:
            video_url: YouTube video URL
            language: Preferred language code
            include_auto_captions: Whether to fall back to auto-generated captions

        Returns:
            Dictionary with transcript text and metadata
        """
        video_id = self.extract_video_id(video_url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {video_url}")
            return None

        try:
            # Try to import youtube_transcript_api
            from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

            # Try to get manual transcript first, then auto-generated
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            transcript = None
            is_auto_generated = False

            try:
                # Try manual transcript
                transcript = transcript_list.find_transcript([language])
            except NoTranscriptFound:
                if include_auto_captions:
                    try:
                        # Fall back to auto-generated
                        transcript = transcript_list.find_generated_transcript([language])
                        is_auto_generated = True
                    except NoTranscriptFound:
                        logger.warning(f"No transcript available for video {video_id}")
                        return None

            if transcript:
                transcript_data = transcript.fetch()

                # Combine transcript segments into full text
                full_text = " ".join([
                    segment.get("text", "")
                    for segment in transcript_data
                ])

                # Calculate duration
                duration = max([
                    segment.get("start", 0) + segment.get("duration", 0)
                    for segment in transcript_data
                ], default=0)

                return {
                    "video_id": video_id,
                    "video_url": video_url,
                    "transcript": full_text,
                    "segments": transcript_data,
                    "language": transcript.language_code,
                    "is_auto_generated": is_auto_generated,
                    "duration_seconds": int(duration)
                }

        except ImportError:
            logger.warning("youtube_transcript_api not installed. Using mock transcript.")
            # Return mock transcript for testing
            return {
                "video_id": video_id,
                "video_url": video_url,
                "transcript": f"[Mock transcript for video {video_id}]",
                "segments": [],
                "language": language,
                "is_auto_generated": True,
                "duration_seconds": 600
            }

        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
            return None

    async def process_transcript(
        self,
        transcript_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Process raw transcript to improve readability

        Uses LLM to:
        - Fix punctuation and capitalization
        - Remove filler words
        - Add paragraph breaks
        - Correct obvious transcription errors
        """
        raw_text = transcript_data.get("transcript", "")

        if not raw_text or len(raw_text) < 50:
            return raw_text

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert transcript editor.
Clean up this auto-generated transcript while preserving all educational content.

Rules:
1. Fix punctuation and capitalization
2. Remove excessive filler words (um, uh, like)
3. Add paragraph breaks at topic transitions
4. Preserve technical terms exactly
5. Do not summarize or remove content
6. Keep the same educational information"""),
            ("human", """Process this transcript:

{transcript}

Output the cleaned transcript:""")
        ])

        try:
            # Process in chunks if too long
            max_chunk = 6000  # tokens approximately
            if len(raw_text) > max_chunk * 4:
                # Split into chunks
                chunks = [raw_text[i:i+max_chunk*4] for i in range(0, len(raw_text), max_chunk*4)]
                processed_chunks = []

                for chunk in chunks:
                    messages = prompt.format_messages(transcript=chunk[:max_chunk*4])
                    response = await self.llm.ainvoke(messages)
                    processed_chunks.append(response.content)

                return "\n\n".join(processed_chunks)
            else:
                messages = prompt.format_messages(transcript=raw_text)
                response = await self.llm.ainvoke(messages)
                return response.content

        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            return raw_text


class TranscriptQualityFilter:
    """
    Filter to assess educational quality of transcripts

    Research basis: Pre-filtering prevents "garbage in, garbage out"
    in the RAG system by scoring content before ingestion.

    Scoring criteria:
    - Information density: Educational content per word
    - Coherence: Logical flow and structure
    - Pedagogical value: Teaching effectiveness
    - Technical accuracy: Correct terminology usage
    - Engagement: Ability to hold learner attention
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, min_score: float = 0.6):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.min_score = min_score

    async def assess_quality(
        self,
        text: str,
        expected_topic: Optional[str] = None
    ) -> ContentQualityScore:
        """
        Assess the educational quality of content

        Args:
            text: Content text to assess
            expected_topic: Expected topic for relevance scoring

        Returns:
            ContentQualityScore with detailed assessment
        """
        if not text or len(text) < 100:
            return ContentQualityScore(
                overall_score=0.0,
                information_density=0.0,
                coherence=0.0,
                pedagogical_value=0.0,
                technical_accuracy=0.0,
                engagement_level=0.0,
                is_suitable=False,
                issues=["Content too short for assessment"]
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational content evaluator.
Assess this content for use in an adaptive learning platform.

Score each criterion from 0.0 to 1.0:
- information_density: Educational content per word (not filler, not tangents)
- coherence: Logical flow, clear structure, smooth transitions
- pedagogical_value: Teaching effectiveness, explanations, examples
- technical_accuracy: Correct terminology, accurate information
- engagement_level: Interesting, accessible, maintains attention

Output JSON:
{{
    "information_density": 0.0-1.0,
    "coherence": 0.0-1.0,
    "pedagogical_value": 0.0-1.0,
    "technical_accuracy": 0.0-1.0,
    "engagement_level": 0.0-1.0,
    "issues": ["list of problems found"],
    "recommendations": ["list of how content could be used/improved"]
}}"""),
            ("human", """Expected Topic: {topic}

Content to assess (first 2000 chars):
{content}

Provide your assessment as JSON:""")
        ])

        try:
            messages = prompt.format_messages(
                topic=expected_topic or "General educational content",
                content=text[:2000]
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

            import json
            assessment = json.loads(response_text.strip())

            # Calculate overall score (weighted average)
            weights = {
                "information_density": 0.2,
                "coherence": 0.15,
                "pedagogical_value": 0.35,
                "technical_accuracy": 0.2,
                "engagement_level": 0.1
            }

            overall_score = sum(
                assessment.get(k, 0.5) * w
                for k, w in weights.items()
            )

            return ContentQualityScore(
                overall_score=round(overall_score, 2),
                information_density=assessment.get("information_density", 0.5),
                coherence=assessment.get("coherence", 0.5),
                pedagogical_value=assessment.get("pedagogical_value", 0.5),
                technical_accuracy=assessment.get("technical_accuracy", 0.5),
                engagement_level=assessment.get("engagement_level", 0.5),
                is_suitable=overall_score >= self.min_score,
                issues=assessment.get("issues", []),
                recommendations=assessment.get("recommendations", [])
            )

        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            # Return neutral score on error
            return ContentQualityScore(
                overall_score=0.5,
                information_density=0.5,
                coherence=0.5,
                pedagogical_value=0.5,
                technical_accuracy=0.5,
                engagement_level=0.5,
                is_suitable=False,
                issues=[f"Assessment error: {str(e)}"]
            )


class PrerequisiteExtractor:
    """
    Extract prerequisite relationships from educational content

    Research basis: Zero-shot LLM prerequisite inference can approach
    expert-level performance when given taxonomic constraints.
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.3)

    async def extract_prerequisites(
        self,
        text: str,
        concepts: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Extract prerequisite relationships from text

        Args:
            text: Educational content
            concepts: List of concepts to analyze

        Returns:
            List of (prerequisite, concept, confidence) tuples
        """
        if not concepts or len(concepts) < 2:
            return []

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying prerequisite relationships in educational content.

For each pair of concepts, determine if one is a prerequisite for the other.
A is a prerequisite for B if understanding A is necessary to understand B.

Output JSON array of relationships:
[
    {{"prerequisite": "Concept A", "concept": "Concept B", "confidence": 0.8, "evidence": "brief reason"}}
]

Only include relationships with confidence > 0.6.
Be conservative - only identify clear, logical prerequisites."""),
            ("human", """Concepts to analyze: {concepts}

Content excerpt:
{content}

Identify prerequisite relationships:""")
        ])

        try:
            messages = prompt.format_messages(
                concepts=", ".join(concepts[:20]),
                content=text[:3000]
            )

            response = await self.llm.ainvoke(messages)

            # Parse JSON
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            import json
            relationships = json.loads(response_text.strip())

            return [
                (r["prerequisite"], r["concept"], r.get("confidence", 0.7))
                for r in relationships
                if r.get("prerequisite") and r.get("concept")
            ]

        except Exception as e:
            logger.error(f"Error extracting prerequisites: {e}")
            return []


class ContentAggregationPipeline:
    """
    Main content aggregation pipeline

    Orchestrates:
    1. Content sourcing (YouTube, articles, etc.)
    2. Quality filtering
    3. Prerequisite extraction
    4. Content matching to Learning Outcomes
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.youtube_agent = YouTubeTranscriptAgent(self.llm)
        self.quality_filter = TranscriptQualityFilter(self.llm)
        self.prerequisite_extractor = PrerequisiteExtractor(self.llm)

    async def aggregate_youtube_video(
        self,
        video_url: str,
        expected_topic: Optional[str] = None
    ) -> Optional[AggregatedContent]:
        """
        Aggregate content from a YouTube video

        Args:
            video_url: YouTube video URL
            expected_topic: Expected topic for relevance filtering

        Returns:
            AggregatedContent if successful and passes quality filter
        """
        import uuid

        # Fetch transcript
        logger.info(f"Fetching transcript for: {video_url}")
        transcript_data = await self.youtube_agent.fetch_transcript(video_url)

        if not transcript_data:
            logger.warning(f"No transcript available for: {video_url}")
            return None

        raw_text = transcript_data.get("transcript", "")

        # Process transcript
        logger.info("Processing transcript...")
        processed_text = await self.youtube_agent.process_transcript(transcript_data)

        # Assess quality
        logger.info("Assessing content quality...")
        quality_score = await self.quality_filter.assess_quality(
            processed_text or raw_text,
            expected_topic
        )

        if not quality_score.is_suitable:
            logger.info(f"Content did not pass quality filter: {quality_score.overall_score}")
            # Still return content but mark as unsuitable
            pass

        return AggregatedContent(
            id=str(uuid.uuid4()),
            source_url=video_url,
            content_type=ContentType.VIDEO,
            title=f"YouTube Video: {transcript_data.get('video_id')}",
            raw_text=raw_text,
            processed_text=processed_text,
            quality_score=quality_score,
            duration_seconds=transcript_data.get("duration_seconds"),
            metadata={
                "video_id": transcript_data.get("video_id"),
                "language": transcript_data.get("language"),
                "is_auto_generated": transcript_data.get("is_auto_generated")
            }
        )

    async def match_content_to_learning_outcomes(
        self,
        content: AggregatedContent,
        learning_outcomes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Match content to Learning Outcomes

        Args:
            content: Aggregated content
            learning_outcomes: List of LOs from Refiner

        Returns:
            Dictionary mapping LO IDs to relevance scores
        """
        text = content.processed_text or content.raw_text

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at matching educational content to learning outcomes.

For each learning outcome, rate how well the content addresses it (0.0-1.0):
- 0.0: Not relevant at all
- 0.5: Partially relevant
- 1.0: Directly addresses the LO

Output JSON: {{"lo_id": relevance_score, ...}}"""),
            ("human", """Learning Outcomes:
{los}

Content excerpt:
{content}

Match scores:""")
        ])

        try:
            lo_descriptions = "\n".join([
                f"- {lo.get('lo_id', 'unknown')}: {lo.get('statement', '')}"
                for lo in learning_outcomes[:10]
            ])

            messages = prompt.format_messages(
                los=lo_descriptions,
                content=text[:2000]
            )

            response = await self.llm.ainvoke(messages)

            # Parse JSON
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            import json
            return json.loads(response_text.strip())

        except Exception as e:
            logger.error(f"Error matching content to LOs: {e}")
            return {}


# Lazy-initialized global instance
_content_pipeline: Optional[ContentAggregationPipeline] = None


def get_content_pipeline() -> ContentAggregationPipeline:
    """Get or create the content pipeline singleton (lazy initialization)"""
    global _content_pipeline
    if _content_pipeline is None:
        _content_pipeline = ContentAggregationPipeline()
    return _content_pipeline
