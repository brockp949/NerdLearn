"""
Multi-Modal Content Transformation Package

Research alignment:
- Podcastfy: Text-to-podcast synthesis with multi-speaker diarization
- React Flow: Interactive diagram generation
- Content Morphing: Seamless modality switching with persistent state

This package enables the "Universal Translator for Complexity" vision -
transforming educational content across modalities while preserving
the learner's conceptual understanding state.
"""
from .podcast_generator import (
    PodcastGenerator,
    PodcastScript,
    PodcastEpisode,
    SpeakerRole,
    get_podcast_generator
)
from .diagram_generator import (
    DiagramGenerator,
    DiagramType,
    DiagramData,
    get_diagram_generator
)
from .content_morpher import (
    ContentMorpher,
    ContentModality,
    MorphedContent,
    ConceptualState,
    get_content_morpher
)

__all__ = [
    # Podcast Generation
    "PodcastGenerator",
    "PodcastScript",
    "PodcastEpisode",
    "SpeakerRole",
    "get_podcast_generator",
    # Diagram Generation
    "DiagramGenerator",
    "DiagramType",
    "DiagramData",
    "get_diagram_generator",
    # Content Morphing
    "ContentMorpher",
    "ContentModality",
    "MorphedContent",
    "ConceptualState",
    "get_content_morpher",
]
