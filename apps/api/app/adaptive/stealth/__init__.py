"""
Stealth Assessment System

Includes:
- TelemetryCollector: Base telemetry collection
- ECD Framework: Evidence-Centered Design assessment
  - Competency Model: What we measure
  - Task Model: Tasks that elicit evidence
  - Evidence Model: How to interpret evidence
  - Assembly Model: How to combine evidence
"""
from .telemetry_collector import (
    TelemetryCollector,
    TelemetryEvent,
    TelemetryEventType,
    EvidenceRule,
    DwellTimeRule,
    VideoEngagementRule,
    ChatQueryRule,
)

from .ecd_framework import (
    # Competency Model
    CompetencyLevel,
    KnowledgeType,
    Competency,
    CompetencyModel,
    # Task Model
    TaskDifficulty,
    TaskType,
    TaskFeatures,
    TaskModel,
    TaskModelRegistry,
    # Evidence Model
    EvidenceObservation,
    EvidenceAccumulator,
    EvidenceRule_ECD,
    ContentEngagementEvidenceRule,
    QuerySophisticationEvidenceRule,
    ProblemSolvingEvidenceRule,
    # Assembly Model
    CompetencyClaim,
    AssemblyModel,
    # Integrated Assessor
    ECDAssessor,
)

from .ml_evidence_rules import (
    # Feature Engineering
    EngagementFeatures,
    FeatureExtractor,
    # Neural Evidence Predictor
    NeuralEvidencePredictor,
    # ML Evidence Rules
    MLEvidenceRule,
    EnsembleEvidencePredictor,
)

__all__ = [
    # Telemetry Collector
    "TelemetryCollector",
    "TelemetryEvent",
    "TelemetryEventType",
    "EvidenceRule",
    "DwellTimeRule",
    "VideoEngagementRule",
    "ChatQueryRule",
    # Competency Model
    "CompetencyLevel",
    "KnowledgeType",
    "Competency",
    "CompetencyModel",
    # Task Model
    "TaskDifficulty",
    "TaskType",
    "TaskFeatures",
    "TaskModel",
    "TaskModelRegistry",
    # Evidence Model
    "EvidenceObservation",
    "EvidenceAccumulator",
    "EvidenceRule_ECD",
    "ContentEngagementEvidenceRule",
    "QuerySophisticationEvidenceRule",
    "ProblemSolvingEvidenceRule",
    # Assembly Model
    "CompetencyClaim",
    "AssemblyModel",
    # Integrated Assessor
    "ECDAssessor",
    # ML Evidence Rules
    "EngagementFeatures",
    "FeatureExtractor",
    "NeuralEvidencePredictor",
    "MLEvidenceRule",
    "EnsembleEvidencePredictor",
]
