"""
Adaptive Learning Engine
Implements FSRS, stealth assessment, Bayesian knowledge tracing, ZPD, Cognitive Load Theory,
Interleaved Practice Scheduling, Deep Knowledge Tracing, and Evidence-Centered Design

Research-aligned implementations:
- FSRS (Free Spaced Repetition Scheduler) - Modern spaced repetition with per-user optimization
- BKT (Bayesian Knowledge Tracing) - Probabilistic mastery tracking (AUC ~0.75)
- DKT (Deep Knowledge Tracing) - LSTM/Transformer models (AUC ~0.83)
- ZPD (Zone of Proximal Development) - Optimal difficulty regulation
- Cognitive Load Theory - Expertise detection, scaffolding fading, real-time estimation
- Stealth Assessment + ECD Framework - Evidence-Centered Design with Task/Assembly models
- Interleaved Practice - g=0.42 effect size, hybrid scheduling for retention
"""

# FSRS with Parameter Learning
from .fsrs import (
    FSRSAlgorithm,
    FSRSCard,
    Rating,
    # Parameter Learning
    ReviewRecord,
    OptimizationResult,
    UserLearningProfile,
    LossFunction,
    FSRSParameterLearner,
    AdaptiveFSRS,
)

# Bayesian Knowledge Tracing
from .bkt import BayesianKnowledgeTracer

# Deep Knowledge Tracing (DKT/Transformer)
from .dkt import (
    DeepKnowledgeTracer,
    DKTLite,
    DKTConfig,
    InteractionSequence,
    KnowledgeState,
)

# Zone of Proximal Development
from .zpd import ZPDRegulator

# Cognitive Load Theory
from .cognitive_load import (
    CognitiveLoadEstimator,
    CognitiveLoadLevel,
    ExpertiseLevel,
    CognitiveLoadEstimate,
    ScaffoldingRecommendation,
)

# Interleaved Practice
from .interleaved import (
    InterleavedScheduler,
    PracticeMode,
    PracticeSequence,
    PracticeItem,
    ConceptProficiency,
)

# Stealth Assessment with ECD Framework
from .stealth import (
    # Base Telemetry
    TelemetryCollector,
    TelemetryEvent,
    TelemetryEventType,
    EvidenceRule,
    # ECD Competency Model
    CompetencyLevel,
    KnowledgeType,
    Competency,
    CompetencyModel,
    # ECD Task Model
    TaskDifficulty,
    TaskType,
    TaskFeatures,
    TaskModel,
    TaskModelRegistry,
    # ECD Evidence Model
    EvidenceObservation,
    EvidenceAccumulator,
    # ECD Assembly Model
    CompetencyClaim,
    AssemblyModel,
    # Integrated ECD Assessor
    ECDAssessor,
)

__all__ = [
    # FSRS
    "FSRSAlgorithm",
    "FSRSCard",
    "Rating",
    # FSRS Parameter Learning
    "ReviewRecord",
    "OptimizationResult",
    "UserLearningProfile",
    "LossFunction",
    "FSRSParameterLearner",
    "AdaptiveFSRS",
    # BKT
    "BayesianKnowledgeTracer",
    # DKT
    "DeepKnowledgeTracer",
    "DKTLite",
    "DKTConfig",
    "InteractionSequence",
    "KnowledgeState",
    # ZPD
    "ZPDRegulator",
    # Cognitive Load
    "CognitiveLoadEstimator",
    "CognitiveLoadLevel",
    "ExpertiseLevel",
    "CognitiveLoadEstimate",
    "ScaffoldingRecommendation",
    # Interleaved Practice
    "InterleavedScheduler",
    "PracticeMode",
    "PracticeSequence",
    "PracticeItem",
    "ConceptProficiency",
    # Stealth Assessment
    "TelemetryCollector",
    "TelemetryEvent",
    "TelemetryEventType",
    "EvidenceRule",
    # ECD Framework
    "CompetencyLevel",
    "KnowledgeType",
    "Competency",
    "CompetencyModel",
    "TaskDifficulty",
    "TaskType",
    "TaskFeatures",
    "TaskModel",
    "TaskModelRegistry",
    "EvidenceObservation",
    "EvidenceAccumulator",
    "CompetencyClaim",
    "AssemblyModel",
    "ECDAssessor",
]
