"""
Generative UI and Adaptive Content Registry

Implements Server-Driven UI (SDUI) architecture for micro-personalization
of the learning interface based on real-time learner state.

Key Components:
1. Atomic Content Registry: LOM-extended granular content units
2. SDUI Schema Generation: JSON UI schemas for client rendering
3. Adaptive Card Assembly: Context-aware interface composition

Content is delivered not through static pages, but via dynamically assembled
schemas where atomic content units are composed based on:
- Selected modality from bandit
- Current scaffolding level
- User fatigue/cognitive load
- Device capabilities

References:
- IEEE Learning Object Metadata (LOM) standard
- Microsoft Adaptive Cards specification
- Apollo GraphQL SDUI patterns
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class ContentUnitType(str, Enum):
    """Types of atomic content units"""
    TEXT_SUMMARY = "text_summary"
    TEXT_EXPLANATION = "text_explanation"
    TEXT_DEFINITION = "text_definition"
    TEXT_EXAMPLE = "text_example"
    VIDEO_CLIP = "video_clip"
    VIDEO_ANIMATION = "video_animation"
    AUDIO_NARRATION = "audio_narration"
    AUDIO_PODCAST = "audio_podcast"
    DIAGRAM_SVG = "diagram_svg"
    DIAGRAM_INTERACTIVE = "diagram_interactive"
    INTERACTION_QUIZ = "interaction_quiz"
    INTERACTION_SIM = "interaction_sim"
    INTERACTION_CODE = "interaction_code"
    SCAFFOLD_HINT = "scaffold_hint"
    SCAFFOLD_BREAKDOWN = "scaffold_breakdown"


class AdaptivityType(str, Enum):
    """Adaptivity tags for LOM extension"""
    SCAFFOLD = "scaffold"  # Provides scaffolding support
    DEEP_DIVE = "deep_dive"  # Advanced exploration
    SUMMARY = "summary"  # Quick overview
    PRACTICE = "practice"  # Active recall
    VISUAL = "visual"  # Visual representation
    AUDIO = "audio"  # Audio representation


class CognitiveLoadLevel(str, Enum):
    """Cognitive load requirements"""
    LOW = "low"  # Passive consumption
    MEDIUM = "medium"  # Active reading/watching
    HIGH = "high"  # Interactive, requires focus


class ScaffoldingLevel(str, Enum):
    """Scaffolding intensity levels"""
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    INTENSIVE = "intensive"


class LayoutType(str, Enum):
    """UI layout patterns"""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    HERO_MEDIA = "hero_media"
    SPLIT_VIEW = "split_view"
    TABBED = "tabbed"
    CAROUSEL = "carousel"


# ============================================================================
# ATOMIC CONTENT UNIT
# ============================================================================

@dataclass
class AtomicContentUnit:
    """
    Granular content component following extended LOM standard

    Instead of storing full "Lessons", we store atomic units that can be
    dynamically assembled based on learner needs.
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    concept_id: str = ""
    parent_content_id: str = ""

    # LOM Core
    title: str = ""
    description: str = ""
    content_type: ContentUnitType = ContentUnitType.TEXT_SUMMARY

    # Content
    content: str = ""  # Markdown, HTML, or reference URL
    content_url: Optional[str] = None  # For media content
    thumbnail_url: Optional[str] = None

    # LOM Educational
    difficulty: float = 0.5  # 0-1 scale
    reading_level: float = 0.5  # Grade level normalized
    typical_duration_seconds: int = 60
    interactivity_level: float = 0.0  # 0=passive, 1=highly interactive

    # LOM Extension: Adaptivity Schema
    adaptivity_type: AdaptivityType = AdaptivityType.SUMMARY
    cognitive_load: CognitiveLoadLevel = CognitiveLoadLevel.MEDIUM
    fatigue_cost: float = 0.2  # How much this depletes user energy
    prerequisites: List[str] = field(default_factory=list)

    # Metadata
    language: str = "en"
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            "id": self.id,
            "concept_id": self.concept_id,
            "parent_content_id": self.parent_content_id,
            "title": self.title,
            "description": self.description,
            "content_type": self.content_type.value,
            "content": self.content,
            "content_url": self.content_url,
            "thumbnail_url": self.thumbnail_url,
            "difficulty": self.difficulty,
            "reading_level": self.reading_level,
            "typical_duration_seconds": self.typical_duration_seconds,
            "interactivity_level": self.interactivity_level,
            "adaptivity_type": self.adaptivity_type.value,
            "cognitive_load": self.cognitive_load.value,
            "fatigue_cost": self.fatigue_cost,
            "prerequisites": self.prerequisites,
            "language": self.language,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicContentUnit":
        """Create from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            concept_id=data.get("concept_id", ""),
            parent_content_id=data.get("parent_content_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            content_type=ContentUnitType(data.get("content_type", "text_summary")),
            content=data.get("content", ""),
            content_url=data.get("content_url"),
            thumbnail_url=data.get("thumbnail_url"),
            difficulty=data.get("difficulty", 0.5),
            reading_level=data.get("reading_level", 0.5),
            typical_duration_seconds=data.get("typical_duration_seconds", 60),
            interactivity_level=data.get("interactivity_level", 0.0),
            adaptivity_type=AdaptivityType(data.get("adaptivity_type", "summary")),
            cognitive_load=CognitiveLoadLevel(data.get("cognitive_load", "medium")),
            fatigue_cost=data.get("fatigue_cost", 0.2),
            prerequisites=data.get("prerequisites", []),
            language=data.get("language", "en"),
            keywords=data.get("keywords", []),
            version=data.get("version", "1.0"),
        )


# ============================================================================
# ADAPTIVE CARD (SDUI Component)
# ============================================================================

@dataclass
class CardElement:
    """Base element in an Adaptive Card"""
    type: str  # "TextBlock", "Image", "Video", "Container", etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List["CardElement"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type, **self.properties}
        if self.children:
            result["items"] = [c.to_dict() for c in self.children]
        return result


@dataclass
class AdaptiveCard:
    """
    Adaptive Card schema for SDUI

    Based on Microsoft Adaptive Cards specification.
    Client renders this JSON schema to display content.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema: str = "http://adaptivecards.io/schemas/adaptive-card.json"
    version: str = "1.5"
    lang: str = "en"

    # Card body
    body: List[CardElement] = field(default_factory=list)

    # Actions (buttons, links)
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Style
    background_color: Optional[str] = None
    min_height: str = "100px"
    vertical_content_alignment: str = "Top"

    # Metadata for adaptation
    modality: str = "text"
    scaffolding_level: ScaffoldingLevel = ScaffoldingLevel.NONE
    fatigue_level_target: float = 0.5
    device_target: str = "all"  # "desktop", "mobile", "tablet", "all"

    def add_text_block(
        self,
        text: str,
        size: str = "Medium",
        weight: str = "Default",
        wrap: bool = True,
        **kwargs
    ):
        """Add a text block to the card"""
        element = CardElement(
            type="TextBlock",
            properties={
                "text": text,
                "size": size,
                "weight": weight,
                "wrap": wrap,
                **kwargs
            }
        )
        self.body.append(element)

    def add_image(
        self,
        url: str,
        alt_text: str = "",
        size: str = "Auto",
        **kwargs
    ):
        """Add an image to the card"""
        element = CardElement(
            type="Image",
            properties={
                "url": url,
                "altText": alt_text,
                "size": size,
                **kwargs
            }
        )
        self.body.append(element)

    def add_media(
        self,
        sources: List[Dict[str, str]],
        poster: Optional[str] = None,
        **kwargs
    ):
        """Add media (video/audio) to the card"""
        props = {"sources": sources, **kwargs}
        if poster:
            props["poster"] = poster
        element = CardElement(type="Media", properties=props)
        self.body.append(element)

    def add_container(
        self,
        items: List[CardElement],
        style: str = "default",
        **kwargs
    ):
        """Add a container with nested elements"""
        element = CardElement(
            type="Container",
            properties={"style": style, **kwargs},
            children=items
        )
        self.body.append(element)

    def add_action(
        self,
        action_type: str,
        title: str,
        **kwargs
    ):
        """Add an action button"""
        action = {"type": action_type, "title": title, **kwargs}
        self.actions.append(action)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "type": "AdaptiveCard",
            "$schema": self.schema,
            "version": self.version,
            "lang": self.lang,
            "body": [e.to_dict() for e in self.body],
            "actions": self.actions,
            "minHeight": self.min_height,
            "verticalContentAlignment": self.vertical_content_alignment,
            "metadata": {
                "id": self.id,
                "modality": self.modality,
                "scaffolding_level": self.scaffolding_level.value,
                "fatigue_level_target": self.fatigue_level_target,
                "device_target": self.device_target,
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# SDUI SCHEMA
# ============================================================================

@dataclass
class SDUISchema:
    """
    Complete SDUI schema for a learning view

    Combines multiple Adaptive Cards with layout instructions
    for the client to render.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    layout: LayoutType = LayoutType.SINGLE_COLUMN

    # Content cards
    primary_content: Optional[AdaptiveCard] = None
    secondary_content: Optional[AdaptiveCard] = None
    scaffolding_content: Optional[AdaptiveCard] = None
    navigation: Optional[AdaptiveCard] = None

    # Metadata
    concept_id: str = ""
    modality: str = "text"
    scaffolding_level: ScaffoldingLevel = ScaffoldingLevel.NONE
    device_optimized: str = "desktop"

    # Adaptation signals
    fatigue_level: float = 0.0
    cognitive_load_estimate: float = 0.5
    estimated_duration_seconds: int = 300

    # Client hints
    font_size_modifier: float = 1.0  # Scale factor for text
    contrast_mode: str = "default"  # "default", "high_contrast", "dark"
    animation_enabled: bool = True
    auto_scroll: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for client"""
        result = {
            "id": self.id,
            "layout": self.layout.value,
            "metadata": {
                "concept_id": self.concept_id,
                "modality": self.modality,
                "scaffolding_level": self.scaffolding_level.value,
                "device_optimized": self.device_optimized,
                "fatigue_level": self.fatigue_level,
                "cognitive_load_estimate": self.cognitive_load_estimate,
                "estimated_duration_seconds": self.estimated_duration_seconds,
            },
            "client_hints": {
                "font_size_modifier": self.font_size_modifier,
                "contrast_mode": self.contrast_mode,
                "animation_enabled": self.animation_enabled,
                "auto_scroll": self.auto_scroll,
            },
            "content": {}
        }

        if self.primary_content:
            result["content"]["primary"] = self.primary_content.to_dict()
        if self.secondary_content:
            result["content"]["secondary"] = self.secondary_content.to_dict()
        if self.scaffolding_content:
            result["content"]["scaffolding"] = self.scaffolding_content.to_dict()
        if self.navigation:
            result["content"]["navigation"] = self.navigation.to_dict()

        return result


# ============================================================================
# GENERATIVE UI REGISTRY
# ============================================================================

class GenerativeUIRegistry:
    """
    Registry and generator for adaptive UI content

    Stores atomic content units and assembles them into
    SDUI schemas based on learner context.
    """

    def __init__(self):
        # Content storage (in production, would be database)
        self.content_units: Dict[str, AtomicContentUnit] = {}

        # Index by concept
        self.concept_index: Dict[str, List[str]] = {}

        # Index by type
        self.type_index: Dict[ContentUnitType, List[str]] = {
            t: [] for t in ContentUnitType
        }

        # Index by adaptivity
        self.adaptivity_index: Dict[AdaptivityType, List[str]] = {
            t: [] for t in AdaptivityType
        }

        logger.info("GenerativeUIRegistry initialized")

    def register_unit(self, unit: AtomicContentUnit):
        """Register an atomic content unit"""
        self.content_units[unit.id] = unit

        # Update indices
        if unit.concept_id:
            if unit.concept_id not in self.concept_index:
                self.concept_index[unit.concept_id] = []
            self.concept_index[unit.concept_id].append(unit.id)

        self.type_index[unit.content_type].append(unit.id)
        self.adaptivity_index[unit.adaptivity_type].append(unit.id)

    def get_units_for_concept(
        self,
        concept_id: str,
        content_type: Optional[ContentUnitType] = None,
        adaptivity_type: Optional[AdaptivityType] = None,
        max_difficulty: Optional[float] = None,
        max_cognitive_load: Optional[CognitiveLoadLevel] = None,
    ) -> List[AtomicContentUnit]:
        """Query content units for a concept with filters"""
        unit_ids = self.concept_index.get(concept_id, [])
        units = [self.content_units[uid] for uid in unit_ids if uid in self.content_units]

        # Apply filters
        if content_type:
            units = [u for u in units if u.content_type == content_type]

        if adaptivity_type:
            units = [u for u in units if u.adaptivity_type == adaptivity_type]

        if max_difficulty is not None:
            units = [u for u in units if u.difficulty <= max_difficulty]

        if max_cognitive_load:
            load_order = [CognitiveLoadLevel.LOW, CognitiveLoadLevel.MEDIUM, CognitiveLoadLevel.HIGH]
            max_idx = load_order.index(max_cognitive_load)
            units = [u for u in units if load_order.index(u.cognitive_load) <= max_idx]

        return units

    def generate_schema(
        self,
        concept_id: str,
        modality: str = "text",
        scaffolding_level: ScaffoldingLevel = ScaffoldingLevel.NONE,
        fatigue_level: float = 0.0,
        device: str = "desktop",
        user_context: Optional[Dict[str, Any]] = None,
    ) -> SDUISchema:
        """
        Generate an SDUI schema for a concept

        Assembles atomic content units based on:
        - Selected modality
        - Scaffolding level
        - User fatigue
        - Device capabilities
        """
        # Determine layout based on device and modality
        if device == "mobile":
            layout = LayoutType.SINGLE_COLUMN
        elif modality == "video":
            layout = LayoutType.HERO_MEDIA
        elif scaffolding_level in [ScaffoldingLevel.MODERATE, ScaffoldingLevel.INTENSIVE]:
            layout = LayoutType.TWO_COLUMN
        else:
            layout = LayoutType.SINGLE_COLUMN

        # Get content units for concept
        units = self.get_units_for_concept(concept_id)

        # Select primary content based on modality
        primary_content = self._create_primary_content(
            units, modality, fatigue_level, device
        )

        # Add scaffolding if needed
        scaffolding_content = None
        if scaffolding_level != ScaffoldingLevel.NONE:
            scaffolding_content = self._create_scaffolding_content(
                units, scaffolding_level, fatigue_level
            )

        # Add secondary content if two-column layout
        secondary_content = None
        if layout == LayoutType.TWO_COLUMN:
            secondary_content = self._create_secondary_content(
                units, modality, primary_content
            )

        # Create navigation
        navigation = self._create_navigation_card(concept_id)

        # Calculate estimated duration
        total_duration = sum(
            u.typical_duration_seconds for u in units
            if u.id in [primary_content.id if primary_content else None]
        ) or 300

        # Apply fatigue-based adjustments
        font_modifier = 1.0
        if fatigue_level > 0.6:
            font_modifier = 1.1  # Slightly larger text when tired

        return SDUISchema(
            layout=layout,
            primary_content=primary_content,
            secondary_content=secondary_content,
            scaffolding_content=scaffolding_content,
            navigation=navigation,
            concept_id=concept_id,
            modality=modality,
            scaffolding_level=scaffolding_level,
            device_optimized=device,
            fatigue_level=fatigue_level,
            cognitive_load_estimate=self._estimate_cognitive_load(units, modality),
            estimated_duration_seconds=total_duration,
            font_size_modifier=font_modifier,
            animation_enabled=fatigue_level < 0.7,  # Disable animations when tired
        )

    def _create_primary_content(
        self,
        units: List[AtomicContentUnit],
        modality: str,
        fatigue_level: float,
        device: str,
    ) -> AdaptiveCard:
        """Create primary content card based on modality"""
        card = AdaptiveCard(modality=modality, device_target=device)

        # Filter units by modality preference
        modality_types = {
            "text": [ContentUnitType.TEXT_EXPLANATION, ContentUnitType.TEXT_SUMMARY],
            "video": [ContentUnitType.VIDEO_CLIP, ContentUnitType.VIDEO_ANIMATION],
            "audio": [ContentUnitType.AUDIO_NARRATION, ContentUnitType.AUDIO_PODCAST],
            "interactive": [ContentUnitType.INTERACTION_QUIZ, ContentUnitType.INTERACTION_SIM],
            "diagram": [ContentUnitType.DIAGRAM_SVG, ContentUnitType.DIAGRAM_INTERACTIVE],
        }

        preferred_types = modality_types.get(modality, modality_types["text"])
        filtered = [u for u in units if u.content_type in preferred_types]

        if not filtered:
            filtered = units  # Fallback to all

        # Sort by fatigue appropriateness
        # When tired, prefer lower cognitive load content
        if fatigue_level > 0.5:
            filtered.sort(key=lambda u: u.fatigue_cost)

        # Build card content
        for unit in filtered[:3]:  # Max 3 units in primary
            if unit.content_type in [ContentUnitType.VIDEO_CLIP, ContentUnitType.VIDEO_ANIMATION]:
                if unit.content_url:
                    card.add_media(
                        sources=[{"mimeType": "video/mp4", "url": unit.content_url}],
                        poster=unit.thumbnail_url,
                    )
                card.add_text_block(unit.description, size="Small")

            elif unit.content_type in [ContentUnitType.DIAGRAM_SVG, ContentUnitType.DIAGRAM_INTERACTIVE]:
                if unit.content_url:
                    card.add_image(unit.content_url, alt_text=unit.title)
                card.add_text_block(unit.description, size="Small")

            else:
                # Text-based content
                card.add_text_block(unit.title, weight="Bolder", size="Large")
                card.add_text_block(unit.content)

        # Add action buttons
        card.add_action("Action.Submit", "Continue", data={"action": "continue"})
        card.add_action("Action.Submit", "Need Help", data={"action": "request_scaffold"})

        return card

    def _create_scaffolding_content(
        self,
        units: List[AtomicContentUnit],
        scaffolding_level: ScaffoldingLevel,
        fatigue_level: float,
    ) -> AdaptiveCard:
        """Create scaffolding support card"""
        card = AdaptiveCard(scaffolding_level=scaffolding_level)

        # Get scaffolding units
        scaffold_units = [u for u in units if u.adaptivity_type == AdaptivityType.SCAFFOLD]

        if scaffolding_level == ScaffoldingLevel.MINIMAL:
            # Just hints
            card.add_text_block("Quick Hints", weight="Bolder")
            hints = [u for u in scaffold_units if u.content_type == ContentUnitType.SCAFFOLD_HINT]
            for hint in hints[:2]:
                card.add_text_block(f"- {hint.content}", size="Small")

        elif scaffolding_level == ScaffoldingLevel.MODERATE:
            # Hints + simplified explanation
            card.add_text_block("Let's Break This Down", weight="Bolder")
            for unit in scaffold_units[:3]:
                card.add_text_block(unit.content)

        elif scaffolding_level == ScaffoldingLevel.INTENSIVE:
            # Full breakdown + examples + maybe diagram
            card.add_text_block("Step-by-Step Guide", weight="Bolder")
            for unit in scaffold_units:
                card.add_text_block(unit.title, weight="Bolder", size="Medium")
                card.add_text_block(unit.content)

                # Add visual if available
                if unit.content_url and "diagram" in unit.content_type.value:
                    card.add_image(unit.content_url)

        # Add action
        card.add_action("Action.Submit", "I Got It!", data={"action": "dismiss_scaffold"})

        return card

    def _create_secondary_content(
        self,
        units: List[AtomicContentUnit],
        modality: str,
        primary: Optional[AdaptiveCard],
    ) -> AdaptiveCard:
        """Create secondary/complementary content"""
        card = AdaptiveCard()
        card.add_text_block("Related Content", weight="Bolder")

        # Get different content types than primary
        if modality == "text":
            # Add diagrams or examples
            diagrams = [u for u in units if u.content_type == ContentUnitType.DIAGRAM_SVG]
            for d in diagrams[:1]:
                if d.content_url:
                    card.add_image(d.content_url, alt_text=d.title)

            examples = [u for u in units if u.content_type == ContentUnitType.TEXT_EXAMPLE]
            for ex in examples[:2]:
                card.add_text_block(ex.title, weight="Bolder", size="Small")
                card.add_text_block(ex.content, size="Small")

        elif modality == "video":
            # Add text summary
            summaries = [u for u in units if u.content_type == ContentUnitType.TEXT_SUMMARY]
            for s in summaries[:1]:
                card.add_text_block("Key Points", weight="Bolder")
                card.add_text_block(s.content)

        return card

    def _create_navigation_card(self, concept_id: str) -> AdaptiveCard:
        """Create navigation card"""
        card = AdaptiveCard()

        card.add_action("Action.Submit", "Previous", data={"action": "nav_prev"})
        card.add_action("Action.Submit", "Next", data={"action": "nav_next"})
        card.add_action("Action.Submit", "Overview", data={"action": "nav_overview"})

        return card

    def _estimate_cognitive_load(
        self,
        units: List[AtomicContentUnit],
        modality: str,
    ) -> float:
        """Estimate cognitive load of content"""
        if not units:
            return 0.5

        # Average difficulty and interactivity
        avg_difficulty = sum(u.difficulty for u in units) / len(units)
        avg_interactivity = sum(u.interactivity_level for u in units) / len(units)

        # Modality modifier
        modality_factor = {
            "text": 0.6,  # Medium - requires active reading
            "video": 0.4,  # Lower - more passive
            "audio": 0.3,  # Lowest - can multitask
            "interactive": 0.9,  # Highest - requires focus
            "diagram": 0.5,
        }.get(modality, 0.5)

        return (avg_difficulty * 0.4 + avg_interactivity * 0.3 + modality_factor * 0.3)

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_units": len(self.content_units),
            "concepts": len(self.concept_index),
            "by_type": {t.value: len(ids) for t, ids in self.type_index.items()},
            "by_adaptivity": {t.value: len(ids) for t, ids in self.adaptivity_index.items()},
        }
