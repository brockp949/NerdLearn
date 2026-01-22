"""
Cognitive Layer Module - Phase 3

Research alignment:
- Affective Computing: Detecting and responding to learner emotions
- Metacognition: Helping learners think about their own learning
- Just-in-Time Interventions: Support when needed most
- Productive Failure: Distinguishing beneficial struggle from frustration

Components:
1. Frustration Detection: Real-time affective state monitoring
2. Metacognition: Confidence ratings, self-explanation, calibration
3. Intervention Engine: Adaptive support orchestration
"""

from .frustration_detector import (
    FrustrationDetector,
    FrustrationLevel,
    StruggleType,
    BehavioralSignal,
    FrustrationEstimate,
    FrustrationIndicators,
    InteractionEvent,
    get_frustration_detector,
)

from .metacognition import (
    MetacognitionPrompter,
    MetacognitionPromptType,
    MetacognitionPrompt,
    CalibrationTracker,
    CalibrationLevel,
    CalibrationData,
    ConfidenceRating,
    SelfExplanation,
    SelfExplanationAnalyzer,
    MetacognitionProfile,
    get_metacognition_prompter,
    get_calibration_tracker,
    get_explanation_analyzer,
)

from .intervention_engine import (
    InterventionEngine,
    InterventionType,
    InterventionPriority,
    Intervention,
    LearnerState,
    InterventionDecision,
    get_intervention_engine,
)

__all__ = [
    # Frustration Detection
    "FrustrationDetector",
    "FrustrationLevel",
    "StruggleType",
    "BehavioralSignal",
    "FrustrationEstimate",
    "FrustrationIndicators",
    "InteractionEvent",
    "get_frustration_detector",
    # Metacognition
    "MetacognitionPrompter",
    "MetacognitionPromptType",
    "MetacognitionPrompt",
    "CalibrationTracker",
    "CalibrationLevel",
    "CalibrationData",
    "ConfidenceRating",
    "SelfExplanation",
    "SelfExplanationAnalyzer",
    "MetacognitionProfile",
    "get_metacognition_prompter",
    "get_calibration_tracker",
    "get_explanation_analyzer",
    # Intervention Engine
    "InterventionEngine",
    "InterventionType",
    "InterventionPriority",
    "Intervention",
    "LearnerState",
    "InterventionDecision",
    "get_intervention_engine",
]
