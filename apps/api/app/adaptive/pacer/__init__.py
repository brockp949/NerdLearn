"""
PACER Learning Protocol Adaptive Module

P.A.C.E.R. Framework for information taxonomy:
- P: Procedural (Practice) - How-to steps and instructions
- A: Analogous (Critique) - Metaphors and comparisons
- C: Conceptual (Mapping) - Theories, principles, causality
- E: Evidence (Store & Rehearse) - Supporting facts/statistics
- R: Reference (Store & Rehearse) - Arbitrary details for SRS

Each type requires distinct cognitive processing actions.
"""

from app.adaptive.pacer.classifier import (
    PACERClassifier,
    ClassificationResult,
    TriageDecision,
)
from app.adaptive.pacer.analogy_engine import (
    AnalogyCritiqueEngine,
    Analogy,
    StructuralMapping,
    BreakdownPoint,
    CritiqueEvaluation,
)
from app.adaptive.pacer.evidence_service import (
    EvidenceLinkingService,
    EvidenceLink,
    LinkedEvidence,
)
from app.adaptive.pacer.procedural_tracker import (
    ProceduralProgressTracker,
    StepResult,
    ProcedureStatus,
)

__all__ = [
    # Classifier
    "PACERClassifier",
    "ClassificationResult",
    "TriageDecision",
    # Analogy Engine
    "AnalogyCritiqueEngine",
    "Analogy",
    "StructuralMapping",
    "BreakdownPoint",
    "CritiqueEvaluation",
    # Evidence Service
    "EvidenceLinkingService",
    "EvidenceLink",
    "LinkedEvidence",
    # Procedural Tracker
    "ProceduralProgressTracker",
    "StepResult",
    "ProcedureStatus",
]
