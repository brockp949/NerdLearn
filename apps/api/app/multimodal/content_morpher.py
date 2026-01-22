"""
Content Morphing Service - Universal Translator for Complexity

Research alignment:
- Content Morphing: Seamless modality switching with persistent state
- Conceptual State Preservation: Maintain understanding across transformations
- Adaptive Modality Selection: Recommend best format for learning context

This is the heart of the multi-modal experience - enabling learners to
switch between text, diagrams, and podcasts while preserving their
conceptual understanding and progress.

Key Features:
1. Modality Detection: Identify current content format
2. Transformation Pipeline: Convert between any supported modalities
3. State Preservation: Track concepts understood in each modality
4. Smart Recommendations: Suggest optimal modality for context
"""
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import hashlib
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .podcast_generator import PodcastGenerator, PodcastEpisode, get_podcast_generator
from .diagram_generator import DiagramGenerator, DiagramData, DiagramType, get_diagram_generator

logger = logging.getLogger(__name__)


class ContentModality(str, Enum):
    """Supported content modalities"""
    TEXT = "text"
    DIAGRAM = "diagram"
    PODCAST = "podcast"
    VIDEO = "video"  # Future
    INTERACTIVE = "interactive"  # Future
    FLASHCARDS = "flashcards"  # Future


class ConceptMasteryLevel(str, Enum):
    """Mastery levels for concepts"""
    UNKNOWN = "unknown"
    INTRODUCED = "introduced"
    FAMILIAR = "familiar"
    UNDERSTOOD = "understood"
    MASTERED = "mastered"


@dataclass
class ConceptState:
    """Tracks understanding of a single concept"""
    concept_id: str
    name: str
    mastery_level: ConceptMasteryLevel = ConceptMasteryLevel.UNKNOWN
    modalities_seen: Set[ContentModality] = field(default_factory=set)
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "mastery_level": self.mastery_level.value,
            "modalities_seen": [m.value for m in self.modalities_seen],
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "interaction_count": self.interaction_count,
            "notes": self.notes
        }


@dataclass
class ConceptualState:
    """
    Persistent state tracking learner's conceptual understanding

    This state persists across modality switches, enabling:
    1. Continuity: Pick up where you left off in any format
    2. Progress Tracking: Know which concepts need more attention
    3. Smart Adaptation: Recommend modalities based on concept state
    """
    user_id: str
    content_id: str
    concepts: Dict[str, ConceptState] = field(default_factory=dict)
    current_modality: ContentModality = ContentModality.TEXT
    modality_history: List[Dict[str, Any]] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.utcnow)
    total_time_seconds: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_concept(self, concept_id: str, name: str) -> ConceptState:
        """Add a new concept to track"""
        if concept_id not in self.concepts:
            self.concepts[concept_id] = ConceptState(
                concept_id=concept_id,
                name=name
            )
        return self.concepts[concept_id]

    def update_concept(
        self,
        concept_id: str,
        modality: ContentModality,
        mastery_delta: int = 0
    ):
        """Update concept state after interaction"""
        if concept_id in self.concepts:
            concept = self.concepts[concept_id]
            concept.modalities_seen.add(modality)
            concept.last_interaction = datetime.utcnow()
            concept.interaction_count += 1

            # Update mastery level
            if mastery_delta > 0:
                levels = list(ConceptMasteryLevel)
                current_idx = levels.index(concept.mastery_level)
                new_idx = min(current_idx + mastery_delta, len(levels) - 1)
                concept.mastery_level = levels[new_idx]

    def switch_modality(self, new_modality: ContentModality):
        """Record modality switch"""
        self.modality_history.append({
            "from": self.current_modality.value,
            "to": new_modality.value,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.current_modality = new_modality

    def get_weak_concepts(self) -> List[ConceptState]:
        """Get concepts that need more attention"""
        weak = []
        for concept in self.concepts.values():
            if concept.mastery_level in [ConceptMasteryLevel.UNKNOWN, ConceptMasteryLevel.INTRODUCED]:
                weak.append(concept)
        return sorted(weak, key=lambda c: c.interaction_count)

    def get_recommended_modality(self, concept_id: str) -> ContentModality:
        """Recommend best modality for a concept based on state"""
        if concept_id not in self.concepts:
            return ContentModality.TEXT

        concept = self.concepts[concept_id]
        seen = concept.modalities_seen

        # If struggling with text, try visual
        if ContentModality.TEXT in seen and concept.mastery_level == ConceptMasteryLevel.INTRODUCED:
            if ContentModality.DIAGRAM not in seen:
                return ContentModality.DIAGRAM

        # If visual learner (based on history)
        diagram_count = sum(1 for h in self.modality_history if h["to"] == "diagram")
        if diagram_count > len(self.modality_history) * 0.4:
            return ContentModality.DIAGRAM

        # If audio preference
        podcast_count = sum(1 for h in self.modality_history if h["to"] == "podcast")
        if podcast_count > len(self.modality_history) * 0.3:
            return ContentModality.PODCAST

        # Default based on mastery
        if concept.mastery_level == ConceptMasteryLevel.MASTERED:
            return ContentModality.TEXT  # Quick review
        elif concept.mastery_level in [ConceptMasteryLevel.FAMILIAR, ConceptMasteryLevel.UNDERSTOOD]:
            return ContentModality.DIAGRAM  # Reinforce connections
        else:
            return ContentModality.PODCAST  # Deep explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "content_id": self.content_id,
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "current_modality": self.current_modality.value,
            "modality_history": self.modality_history,
            "session_start": self.session_start.isoformat(),
            "total_time_seconds": self.total_time_seconds,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptualState":
        """Reconstruct from dictionary"""
        state = cls(
            user_id=data["user_id"],
            content_id=data["content_id"],
            current_modality=ContentModality(data.get("current_modality", "text")),
            modality_history=data.get("modality_history", []),
            total_time_seconds=data.get("total_time_seconds", 0),
            metadata=data.get("metadata", {})
        )

        if "session_start" in data:
            state.session_start = datetime.fromisoformat(data["session_start"])

        for concept_id, concept_data in data.get("concepts", {}).items():
            concept = ConceptState(
                concept_id=concept_data["concept_id"],
                name=concept_data["name"],
                mastery_level=ConceptMasteryLevel(concept_data.get("mastery_level", "unknown")),
                interaction_count=concept_data.get("interaction_count", 0),
                notes=concept_data.get("notes", [])
            )
            concept.modalities_seen = {
                ContentModality(m) for m in concept_data.get("modalities_seen", [])
            }
            if concept_data.get("last_interaction"):
                concept.last_interaction = datetime.fromisoformat(concept_data["last_interaction"])
            state.concepts[concept_id] = concept

        return state


@dataclass
class MorphedContent:
    """Result of content transformation"""
    original_modality: ContentModality
    target_modality: ContentModality
    content: Any  # Type depends on modality
    concepts_extracted: List[str]
    transformation_notes: str
    conceptual_state: Optional[ConceptualState] = None

    def to_dict(self) -> Dict[str, Any]:
        content_serialized = self.content
        if hasattr(self.content, 'to_dict'):
            content_serialized = self.content.to_dict()
        elif hasattr(self.content, 'to_react_flow'):
            content_serialized = self.content.to_react_flow()

        return {
            "original_modality": self.original_modality.value,
            "target_modality": self.target_modality.value,
            "content": content_serialized,
            "concepts_extracted": self.concepts_extracted,
            "transformation_notes": self.transformation_notes,
            "conceptual_state": self.conceptual_state.to_dict() if self.conceptual_state else None
        }


class ContentMorpher:
    """
    Universal content transformer - the heart of multi-modal learning

    Enables seamless switching between:
    - Text: Markdown/HTML educational content
    - Diagrams: Interactive React Flow visualizations
    - Podcasts: Audio explanations with multiple speakers

    Key principle: Conceptual state persists across modality switches,
    so learners can pick up where they left off in any format.
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        podcast_generator: Optional[PodcastGenerator] = None,
        diagram_generator: Optional[DiagramGenerator] = None
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.podcast_gen = podcast_generator or PodcastGenerator(llm=self.llm)
        self.diagram_gen = diagram_generator or DiagramGenerator(llm=self.llm)

        # State cache (in production, use Redis/DB)
        self._state_cache: Dict[str, ConceptualState] = {}

    def _get_state_key(self, user_id: str, content_id: str) -> str:
        """Generate cache key for conceptual state"""
        return f"{user_id}:{content_id}"

    def get_or_create_state(
        self,
        user_id: str,
        content_id: str
    ) -> ConceptualState:
        """Get existing state or create new one"""
        key = self._get_state_key(user_id, content_id)
        if key not in self._state_cache:
            self._state_cache[key] = ConceptualState(
                user_id=user_id,
                content_id=content_id
            )
        return self._state_cache[key]

    def save_state(self, state: ConceptualState):
        """Persist conceptual state"""
        key = self._get_state_key(state.user_id, state.content_id)
        self._state_cache[key] = state

    async def extract_concepts(self, content: str) -> List[Dict[str, str]]:
        """
        Extract key concepts from content using LLM

        Returns:
            List of {"id": str, "name": str, "description": str}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the key educational concepts from this content.
For each concept, provide:
1. A unique ID (snake_case)
2. A name (human-readable)
3. A brief description

Output JSON array:
[
    {"id": "concept_id", "name": "Concept Name", "description": "Brief description"}
]

Extract 3-10 core concepts. Focus on the main ideas being taught."""),
            ("human", "{content}")
        ])

        try:
            messages = prompt.format_messages(content=content[:3000])
            response = await self.llm.ainvoke(messages)

            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            concepts = json.loads(response_text.strip())
            return concepts

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return [{"id": "main_concept", "name": "Main Concept", "description": "Primary topic"}]

    async def morph(
        self,
        content: str,
        source_modality: ContentModality,
        target_modality: ContentModality,
        user_id: Optional[str] = None,
        content_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> MorphedContent:
        """
        Transform content from one modality to another

        Args:
            content: Source content (text, diagram JSON, or audio reference)
            source_modality: Current format
            target_modality: Desired format
            user_id: For state persistence
            content_id: Unique content identifier
            options: Transformation options (duration, diagram_type, etc.)

        Returns:
            MorphedContent with transformed content and updated state
        """
        options = options or {}
        logger.info(f"Morphing content: {source_modality.value} -> {target_modality.value}")

        # Get or create conceptual state
        state = None
        if user_id and content_id:
            state = self.get_or_create_state(user_id, content_id)
            state.switch_modality(target_modality)

        # Extract concepts from source
        text_content = content
        if source_modality == ContentModality.DIAGRAM:
            # Extract text from diagram
            try:
                diagram_data = json.loads(content) if isinstance(content, str) else content
                labels = [n.get("data", {}).get("label", "") for n in diagram_data.get("nodes", [])]
                text_content = " ".join(labels)
            except Exception:
                text_content = str(content)

        concepts = await self.extract_concepts(text_content)
        concept_names = [c["name"] for c in concepts]

        # Update state with concepts
        if state:
            for concept in concepts:
                state.add_concept(concept["id"], concept["name"])
                state.update_concept(concept["id"], target_modality)

        # Perform transformation
        transformed_content = None
        transformation_notes = ""

        if target_modality == ContentModality.TEXT:
            transformed_content, transformation_notes = await self._to_text(
                content, source_modality, concept_names, options
            )

        elif target_modality == ContentModality.DIAGRAM:
            transformed_content, transformation_notes = await self._to_diagram(
                text_content, concept_names, options
            )

        elif target_modality == ContentModality.PODCAST:
            transformed_content, transformation_notes = await self._to_podcast(
                text_content, concept_names, options
            )

        # Save updated state
        if state:
            self.save_state(state)

        return MorphedContent(
            original_modality=source_modality,
            target_modality=target_modality,
            content=transformed_content,
            concepts_extracted=concept_names,
            transformation_notes=transformation_notes,
            conceptual_state=state
        )

    async def _to_text(
        self,
        content: str,
        source_modality: ContentModality,
        concepts: List[str],
        options: Dict[str, Any]
    ) -> tuple:
        """Transform content to text format"""
        if source_modality == ContentModality.TEXT:
            return content, "Content already in text format"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Transform this content into clear, well-structured educational text.

Guidelines:
1. Organize with headings and sections
2. Explain key concepts clearly
3. Use examples where helpful
4. Include transitions between ideas
5. Format as Markdown

Key concepts to cover: {concepts}"""),
            ("human", "Content to transform:\n{content}")
        ])

        try:
            messages = prompt.format_messages(
                concepts=", ".join(concepts),
                content=content[:4000]
            )
            response = await self.llm.ainvoke(messages)
            return response.content, f"Transformed from {source_modality.value} to text"
        except Exception as e:
            logger.error(f"Error transforming to text: {e}")
            return content, f"Error: {str(e)}"

    async def _to_diagram(
        self,
        content: str,
        concepts: List[str],
        options: Dict[str, Any]
    ) -> tuple:
        """Transform content to diagram format"""
        diagram_type = DiagramType(options.get("diagram_type", "concept_map"))

        try:
            diagram = await self.diagram_gen.generate(
                content=content,
                diagram_type=diagram_type,
                title=options.get("title"),
                focus_concepts=concepts[:5]
            )
            return diagram, f"Generated {diagram_type.value} with {len(diagram.nodes)} nodes"
        except Exception as e:
            logger.error(f"Error generating diagram: {e}")
            # Return minimal diagram
            from .diagram_generator import ReactFlowNode, ReactFlowEdge, NodeType
            return DiagramData(
                id="error_diagram",
                type=diagram_type,
                title="Diagram",
                nodes=[ReactFlowNode(
                    id="n1",
                    type=NodeType.DEFAULT,
                    position={"x": 100, "y": 100},
                    data={"label": concepts[0] if concepts else "Content"}
                )],
                edges=[],
                mermaid_source="flowchart TD\n    A[Content]"
            ), f"Error: {str(e)}"

    async def _to_podcast(
        self,
        content: str,
        concepts: List[str],
        options: Dict[str, Any]
    ) -> tuple:
        """Transform content to podcast format"""
        duration = options.get("duration_minutes", 10)
        style = options.get("style", "educational")

        try:
            episode = await self.podcast_gen.generate(
                content=content,
                topic=concepts[0] if concepts else "Topic",
                duration_minutes=duration,
                style=style
            )
            return episode, f"Generated {duration}-minute podcast with {len(episode.script.segments)} segments"
        except Exception as e:
            logger.error(f"Error generating podcast: {e}")
            from .podcast_generator import PodcastEpisode, PodcastScript, ScriptSegment, SpeakerRole
            # Return minimal podcast
            return PodcastEpisode(
                id="error_episode",
                title=concepts[0] if concepts else "Episode",
                script=PodcastScript(
                    title="Episode",
                    segments=[ScriptSegment(
                        speaker=SpeakerRole.HOST,
                        text=content[:500],
                        duration_seconds=60,
                        emotion="neutral"
                    )],
                    total_duration_seconds=60
                ),
                audio_url=None,
                duration_seconds=60
            ), f"Error: {str(e)}"

    async def recommend_modality(
        self,
        user_id: str,
        content_id: str,
        current_concept: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend optimal modality based on state and context

        Args:
            user_id: User identifier
            content_id: Content identifier
            current_concept: Concept being studied
            context: Additional context (time of day, device, etc.)

        Returns:
            Recommendation with reasoning
        """
        state = self.get_or_create_state(user_id, content_id)
        context = context or {}

        # Get weak concepts
        weak_concepts = state.get_weak_concepts()

        # Determine recommendation
        recommended = ContentModality.TEXT
        reason = "Default to text for new content"

        if current_concept and current_concept in state.concepts:
            recommended = state.get_recommended_modality(current_concept)
            concept_state = state.concepts[current_concept]
            reason = f"Based on {concept_state.mastery_level.value} mastery and {len(concept_state.modalities_seen)} modalities tried"

        elif weak_concepts:
            # Focus on weakest concept
            weakest = weak_concepts[0]
            recommended = state.get_recommended_modality(weakest.concept_id)
            reason = f"Focus on weak concept: {weakest.name}"

        # Context adjustments
        if context.get("device") == "mobile":
            if recommended == ContentModality.DIAGRAM:
                # Diagrams harder on mobile
                recommended = ContentModality.PODCAST
                reason += " (adjusted for mobile device)"

        if context.get("time_available_minutes", 60) < 5:
            recommended = ContentModality.TEXT
            reason += " (quick review due to limited time)"

        return {
            "recommended_modality": recommended.value,
            "reason": reason,
            "alternatives": [
                m.value for m in ContentModality
                if m != recommended and m in [ContentModality.TEXT, ContentModality.DIAGRAM, ContentModality.PODCAST]
            ],
            "weak_concepts": [c.name for c in weak_concepts[:3]],
            "current_state": state.current_modality.value
        }

    async def get_learning_summary(
        self,
        user_id: str,
        content_id: str
    ) -> Dict[str, Any]:
        """
        Get summary of learning progress across modalities

        Returns:
            Summary with concept mastery and modality usage
        """
        state = self.get_or_create_state(user_id, content_id)

        # Calculate modality distribution
        modality_counts = {}
        for concept in state.concepts.values():
            for modality in concept.modalities_seen:
                modality_counts[modality.value] = modality_counts.get(modality.value, 0) + 1

        # Calculate mastery distribution
        mastery_counts = {}
        for concept in state.concepts.values():
            level = concept.mastery_level.value
            mastery_counts[level] = mastery_counts.get(level, 0) + 1

        # Calculate overall progress
        total_concepts = len(state.concepts)
        mastered = sum(1 for c in state.concepts.values()
                       if c.mastery_level in [ConceptMasteryLevel.UNDERSTOOD, ConceptMasteryLevel.MASTERED])
        progress = (mastered / total_concepts * 100) if total_concepts > 0 else 0

        return {
            "user_id": user_id,
            "content_id": content_id,
            "total_concepts": total_concepts,
            "progress_percent": round(progress, 1),
            "mastery_distribution": mastery_counts,
            "modality_usage": modality_counts,
            "session_duration_seconds": state.total_time_seconds,
            "modality_switches": len(state.modality_history),
            "weak_concepts": [c.name for c in state.get_weak_concepts()[:5]],
            "current_modality": state.current_modality.value
        }


# Global instance (lazy initialization)
_content_morpher: Optional[ContentMorpher] = None


def get_content_morpher_instance() -> ContentMorpher:
    """Get or create the content morpher instance (lazy)"""
    global _content_morpher
    if _content_morpher is None:
        _content_morpher = ContentMorpher()
    return _content_morpher


async def get_content_morpher() -> ContentMorpher:
    """Dependency injection"""
    return get_content_morpher_instance()
