"""
Mental Model and Misconception Detection System
Identifies and tracks learner misconceptions for targeted remediation

Research basis:
- Mental Model Frameworks in Education: Evidence and Implementation
- Evidence-Centered Design for misconception detection
- Error pattern analysis for adaptive remediation
- Conceptual change theory

Key concepts:
- Mental models: Internal representations of concepts
- Misconceptions: Systematic errors in understanding
- Error patterns: Recurring mistakes indicating misconceptions
- Remediation: Targeted interventions to correct misconceptions

Detection methods:
- Error pattern analysis
- Response time anomalies
- Answer choice analysis (distractor analysis)
- Confidence-accuracy calibration
"""
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math
import statistics


class MisconceptionType(str, Enum):
    """Types of misconceptions based on cognitive research"""
    FACTUAL = "factual"              # Wrong facts memorized
    PROCEDURAL = "procedural"        # Incorrect process/steps
    CONCEPTUAL = "conceptual"        # Misunderstanding core concepts
    OVERGENERALIZATION = "overgeneralization"  # Applying rules too broadly
    UNDERGENERALIZATION = "undergeneralization"  # Not applying rules broadly enough
    PRECONCEPTION = "preconception"  # Prior beliefs conflicting with correct knowledge
    VERNACULAR = "vernacular"        # Confusion from everyday vs technical terms


class MisconceptionSeverity(str, Enum):
    """Severity of misconception impact"""
    LOW = "low"           # Minor, easily corrected
    MEDIUM = "medium"     # Moderate impact on learning
    HIGH = "high"         # Significant barrier to progress
    CRITICAL = "critical"  # Blocks understanding of many concepts


class RemediationStrategy(str, Enum):
    """Strategies for addressing misconceptions"""
    DIRECT_INSTRUCTION = "direct_instruction"    # Explicit teaching of correct concept
    COGNITIVE_CONFLICT = "cognitive_conflict"    # Create situations exposing misconception
    ANALOGICAL_BRIDGING = "analogical_bridging"  # Use familiar concepts to explain
    WORKED_EXAMPLES = "worked_examples"          # Show correct process step by step
    REFUTATIONAL_TEXT = "refutational_text"      # Explicitly address and refute misconception
    SOCRATIC_DIALOGUE = "socratic_dialogue"      # Guide through questioning
    VISUALIZATION = "visualization"              # Visual representations to clarify


@dataclass
class Misconception:
    """A specific misconception"""
    misconception_id: str
    name: str
    description: str
    misconception_type: MisconceptionType
    severity: MisconceptionSeverity
    domain: str  # Subject area
    concept_ids: List[int]  # Related concepts
    prerequisite_for: List[int] = field(default_factory=list)  # Concepts this blocks
    common_triggers: List[str] = field(default_factory=list)  # Situations that trigger
    indicator_patterns: List[str] = field(default_factory=list)  # Error patterns
    remediation_strategies: List[RemediationStrategy] = field(default_factory=list)


@dataclass
class ErrorPattern:
    """A pattern of errors indicating potential misconception"""
    pattern_id: str
    user_id: int
    concept_id: int
    error_type: str
    frequency: int = 0
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)
    associated_misconceptions: List[str] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)  # Response times for these errors
    confidence_ratings: List[float] = field(default_factory=list)  # User's confidence when wrong


@dataclass
class MentalModel:
    """User's mental model for a concept"""
    user_id: int
    concept_id: int
    accuracy_score: float = 0.5  # 0-1, how accurate their model is
    completeness_score: float = 0.5  # 0-1, how complete their model is
    stability_score: float = 0.5  # 0-1, how consistent their understanding is
    identified_misconceptions: List[str] = field(default_factory=list)
    correct_understandings: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    assessment_count: int = 0


@dataclass
class DiagnosticResult:
    """Result of misconception diagnostic"""
    user_id: int
    concept_id: int
    detected_misconceptions: List[Tuple[str, float]]  # (misconception_id, confidence)
    mental_model_accuracy: float
    error_patterns_found: List[ErrorPattern]
    recommended_remediations: List[Dict[str, Any]]
    confidence_calibration: float  # How well user knows what they know
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResponseData:
    """Data from a single response for analysis"""
    question_id: int
    concept_id: int
    is_correct: bool
    selected_answer: str
    correct_answer: str
    response_time_ms: int
    confidence: Optional[float] = None  # 0-1 if collected
    distractor_type: Optional[str] = None  # Type of wrong answer if applicable
    attempt_number: int = 1


class MisconceptionLibrary:
    """
    Library of known misconceptions per domain

    In production, this would be populated from a database
    with domain expert-curated misconceptions
    """

    def __init__(self):
        self.misconceptions: Dict[str, Misconception] = {}
        self._load_common_misconceptions()

    def _load_common_misconceptions(self):
        """Load common educational misconceptions"""
        # Example misconceptions - would come from database in production
        common = [
            Misconception(
                misconception_id="math_001",
                name="Multiplication Always Makes Bigger",
                description="Belief that multiplying always increases a number",
                misconception_type=MisconceptionType.OVERGENERALIZATION,
                severity=MisconceptionSeverity.MEDIUM,
                domain="mathematics",
                concept_ids=[101, 102],  # multiplication, fractions
                indicator_patterns=["multiply_fraction_increase_expected"],
                remediation_strategies=[
                    RemediationStrategy.COGNITIVE_CONFLICT,
                    RemediationStrategy.WORKED_EXAMPLES,
                ],
            ),
            Misconception(
                misconception_id="math_002",
                name="Division Always Makes Smaller",
                description="Belief that dividing always decreases a number",
                misconception_type=MisconceptionType.OVERGENERALIZATION,
                severity=MisconceptionSeverity.MEDIUM,
                domain="mathematics",
                concept_ids=[103, 102],  # division, fractions
                indicator_patterns=["divide_fraction_decrease_expected"],
                remediation_strategies=[
                    RemediationStrategy.COGNITIVE_CONFLICT,
                    RemediationStrategy.VISUALIZATION,
                ],
            ),
            Misconception(
                misconception_id="phys_001",
                name="Heavier Objects Fall Faster",
                description="Belief that mass affects free-fall speed",
                misconception_type=MisconceptionType.PRECONCEPTION,
                severity=MisconceptionSeverity.HIGH,
                domain="physics",
                concept_ids=[201, 202],  # gravity, mechanics
                indicator_patterns=["mass_speed_correlation"],
                remediation_strategies=[
                    RemediationStrategy.COGNITIVE_CONFLICT,
                    RemediationStrategy.VISUALIZATION,
                    RemediationStrategy.REFUTATIONAL_TEXT,
                ],
            ),
            Misconception(
                misconception_id="prog_001",
                name="Variable Assignment Copies Object",
                description="Confusion about reference vs value assignment",
                misconception_type=MisconceptionType.CONCEPTUAL,
                severity=MisconceptionSeverity.HIGH,
                domain="programming",
                concept_ids=[301, 302],  # variables, references
                indicator_patterns=["reference_mutation_surprise"],
                remediation_strategies=[
                    RemediationStrategy.VISUALIZATION,
                    RemediationStrategy.WORKED_EXAMPLES,
                ],
            ),
        ]

        for m in common:
            self.misconceptions[m.misconception_id] = m

    def get_misconception(self, misconception_id: str) -> Optional[Misconception]:
        return self.misconceptions.get(misconception_id)

    def get_by_concept(self, concept_id: int) -> List[Misconception]:
        """Get all misconceptions related to a concept"""
        return [
            m for m in self.misconceptions.values()
            if concept_id in m.concept_ids
        ]

    def get_by_domain(self, domain: str) -> List[Misconception]:
        """Get all misconceptions in a domain"""
        return [
            m for m in self.misconceptions.values()
            if m.domain == domain
        ]

    def add_misconception(self, misconception: Misconception):
        """Add a new misconception to the library"""
        self.misconceptions[misconception.misconception_id] = misconception


class MentalModelDetector:
    """
    Detects misconceptions and assesses mental models

    Uses multiple signals:
    - Error patterns across questions
    - Response time anomalies
    - Distractor analysis (which wrong answers are chosen)
    - Confidence-accuracy calibration
    """

    # Thresholds based on research
    ERROR_PATTERN_THRESHOLD = 3  # Minimum errors to identify pattern
    CONFIDENCE_MISCALIBRATION_THRESHOLD = 0.3  # High confidence but wrong
    RESPONSE_TIME_ANOMALY_MULTIPLIER = 2.0  # Fast wrong answers suggest misconception

    def __init__(self, misconception_library: MisconceptionLibrary = None):
        self.library = misconception_library or MisconceptionLibrary()
        self.user_error_patterns: Dict[int, List[ErrorPattern]] = {}
        self.user_mental_models: Dict[int, Dict[int, MentalModel]] = {}

    def analyze_response(
        self,
        user_id: int,
        response: ResponseData,
        concept_average_time: float = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single response for misconception indicators

        Args:
            user_id: User ID
            response: Response data
            concept_average_time: Average response time for this concept

        Returns:
            Analysis results with potential misconceptions
        """
        signals = []
        misconception_indicators = []

        if not response.is_correct:
            # Signal 1: Distractor analysis
            if response.distractor_type:
                signals.append({
                    "type": "distractor",
                    "value": response.distractor_type,
                    "weight": 0.7,
                })

            # Signal 2: Confidence-accuracy miscalibration
            if response.confidence and response.confidence > 0.7:
                signals.append({
                    "type": "overconfidence",
                    "value": response.confidence,
                    "weight": 0.8,
                })

            # Signal 3: Response time anomaly (fast wrong answer)
            if concept_average_time and response.response_time_ms < concept_average_time / 2:
                signals.append({
                    "type": "fast_error",
                    "value": response.response_time_ms / concept_average_time,
                    "weight": 0.6,
                })

            # Record error pattern
            self._record_error_pattern(user_id, response)

            # Check for matching misconceptions
            concept_misconceptions = self.library.get_by_concept(response.concept_id)
            for misconception in concept_misconceptions:
                if self._matches_misconception_pattern(response, misconception):
                    misconception_indicators.append({
                        "misconception_id": misconception.misconception_id,
                        "name": misconception.name,
                        "confidence": self._calculate_match_confidence(signals),
                    })

        # Update mental model
        self._update_mental_model(user_id, response)

        return {
            "is_correct": response.is_correct,
            "signals": signals,
            "potential_misconceptions": misconception_indicators,
            "requires_intervention": len(misconception_indicators) > 0 and any(
                m["confidence"] > 0.6 for m in misconception_indicators
            ),
        }

    def diagnose_concept(
        self,
        user_id: int,
        concept_id: int,
        responses: List[ResponseData],
    ) -> DiagnosticResult:
        """
        Comprehensive misconception diagnosis for a concept

        Analyzes response history to identify misconceptions
        and assess mental model accuracy
        """
        if not responses:
            return DiagnosticResult(
                user_id=user_id,
                concept_id=concept_id,
                detected_misconceptions=[],
                mental_model_accuracy=0.5,
                error_patterns_found=[],
                recommended_remediations=[],
                confidence_calibration=0.5,
            )

        # Filter to this concept
        concept_responses = [r for r in responses if r.concept_id == concept_id]

        if not concept_responses:
            return DiagnosticResult(
                user_id=user_id,
                concept_id=concept_id,
                detected_misconceptions=[],
                mental_model_accuracy=0.5,
                error_patterns_found=[],
                recommended_remediations=[],
                confidence_calibration=0.5,
            )

        # Calculate accuracy
        correct_count = sum(1 for r in concept_responses if r.is_correct)
        accuracy = correct_count / len(concept_responses)

        # Calculate confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(concept_responses)

        # Find error patterns
        error_patterns = self._find_error_patterns(user_id, concept_id, concept_responses)

        # Detect misconceptions
        detected_misconceptions = self._detect_misconceptions(
            concept_id, concept_responses, error_patterns
        )

        # Generate remediation recommendations
        remediations = self._generate_remediations(detected_misconceptions)

        return DiagnosticResult(
            user_id=user_id,
            concept_id=concept_id,
            detected_misconceptions=detected_misconceptions,
            mental_model_accuracy=accuracy,
            error_patterns_found=error_patterns,
            recommended_remediations=remediations,
            confidence_calibration=confidence_calibration,
        )

    def get_mental_model(
        self,
        user_id: int,
        concept_id: int,
    ) -> MentalModel:
        """Get or create mental model for user/concept"""
        if user_id not in self.user_mental_models:
            self.user_mental_models[user_id] = {}

        if concept_id not in self.user_mental_models[user_id]:
            self.user_mental_models[user_id][concept_id] = MentalModel(
                user_id=user_id,
                concept_id=concept_id,
            )

        return self.user_mental_models[user_id][concept_id]

    def get_user_misconception_summary(
        self,
        user_id: int,
    ) -> Dict[str, Any]:
        """Get summary of all detected misconceptions for a user"""
        if user_id not in self.user_mental_models:
            return {
                "user_id": user_id,
                "total_misconceptions": 0,
                "by_severity": {},
                "by_type": {},
                "concepts_affected": [],
                "recommended_focus": [],
            }

        all_misconceptions: Set[str] = set()
        concepts_affected: Set[int] = set()

        for concept_id, model in self.user_mental_models[user_id].items():
            all_misconceptions.update(model.identified_misconceptions)
            if model.identified_misconceptions:
                concepts_affected.add(concept_id)

        # Categorize by severity and type
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for m_id in all_misconceptions:
            misconception = self.library.get_misconception(m_id)
            if misconception:
                severity = misconception.severity.value
                m_type = misconception.misconception_type.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
                by_type[m_type] = by_type.get(m_type, 0) + 1

        # Prioritize critical and high severity for focus
        recommended_focus = []
        for m_id in all_misconceptions:
            misconception = self.library.get_misconception(m_id)
            if misconception and misconception.severity in [
                MisconceptionSeverity.CRITICAL,
                MisconceptionSeverity.HIGH
            ]:
                recommended_focus.append({
                    "misconception_id": m_id,
                    "name": misconception.name,
                    "severity": misconception.severity.value,
                    "remediation": misconception.remediation_strategies[0].value
                    if misconception.remediation_strategies else "direct_instruction",
                })

        return {
            "user_id": user_id,
            "total_misconceptions": len(all_misconceptions),
            "by_severity": by_severity,
            "by_type": by_type,
            "concepts_affected": list(concepts_affected),
            "recommended_focus": recommended_focus[:5],  # Top 5 priorities
        }

    def _record_error_pattern(self, user_id: int, response: ResponseData):
        """Record an error for pattern analysis"""
        if user_id not in self.user_error_patterns:
            self.user_error_patterns[user_id] = []

        # Look for existing pattern
        existing = None
        for pattern in self.user_error_patterns[user_id]:
            if (pattern.concept_id == response.concept_id and
                pattern.error_type == response.distractor_type):
                existing = pattern
                break

        if existing:
            existing.frequency += 1
            existing.last_occurrence = datetime.now()
            existing.response_times.append(response.response_time_ms)
            if response.confidence:
                existing.confidence_ratings.append(response.confidence)
        else:
            self.user_error_patterns[user_id].append(ErrorPattern(
                pattern_id=f"err_{user_id}_{response.concept_id}_{len(self.user_error_patterns[user_id])}",
                user_id=user_id,
                concept_id=response.concept_id,
                error_type=response.distractor_type or "unknown",
                frequency=1,
                response_times=[response.response_time_ms],
                confidence_ratings=[response.confidence] if response.confidence else [],
            ))

    def _update_mental_model(self, user_id: int, response: ResponseData):
        """Update mental model based on response"""
        model = self.get_mental_model(user_id, response.concept_id)

        # Update accuracy score (exponential moving average)
        alpha = 0.3  # Learning rate
        new_score = 1.0 if response.is_correct else 0.0
        model.accuracy_score = alpha * new_score + (1 - alpha) * model.accuracy_score

        # Update stability score based on consistency
        model.assessment_count += 1
        if model.assessment_count > 3:
            # More consistent responses = higher stability
            recent_variance = abs(new_score - model.accuracy_score)
            model.stability_score = max(0.1, model.stability_score - recent_variance * 0.1)
            model.stability_score = min(1.0, model.stability_score + 0.05)

        model.last_updated = datetime.now()

    def _matches_misconception_pattern(
        self,
        response: ResponseData,
        misconception: Misconception
    ) -> bool:
        """Check if response matches misconception indicators"""
        if response.distractor_type:
            return response.distractor_type in misconception.indicator_patterns
        return False

    def _calculate_match_confidence(self, signals: List[Dict]) -> float:
        """Calculate confidence that a misconception is present"""
        if not signals:
            return 0.0

        total_weight = sum(s["weight"] for s in signals)
        return min(1.0, total_weight / 2)  # Normalize

    def _calculate_confidence_calibration(
        self,
        responses: List[ResponseData]
    ) -> float:
        """
        Calculate how well-calibrated user's confidence is

        Good calibration: confidence matches accuracy
        Poor calibration: high confidence when wrong, low when right
        """
        responses_with_confidence = [r for r in responses if r.confidence is not None]

        if not responses_with_confidence:
            return 0.5  # Unknown

        calibration_errors = []
        for r in responses_with_confidence:
            expected = 1.0 if r.is_correct else 0.0
            error = abs(r.confidence - expected)
            calibration_errors.append(error)

        avg_error = statistics.mean(calibration_errors)
        return 1.0 - avg_error  # Higher = better calibrated

    def _find_error_patterns(
        self,
        user_id: int,
        concept_id: int,
        responses: List[ResponseData]
    ) -> List[ErrorPattern]:
        """Find significant error patterns"""
        if user_id not in self.user_error_patterns:
            return []

        concept_patterns = [
            p for p in self.user_error_patterns[user_id]
            if p.concept_id == concept_id and p.frequency >= self.ERROR_PATTERN_THRESHOLD
        ]

        return concept_patterns

    def _detect_misconceptions(
        self,
        concept_id: int,
        responses: List[ResponseData],
        error_patterns: List[ErrorPattern],
    ) -> List[Tuple[str, float]]:
        """Detect misconceptions from patterns and responses"""
        detected = []

        concept_misconceptions = self.library.get_by_concept(concept_id)

        for misconception in concept_misconceptions:
            confidence = 0.0

            # Check error patterns
            for pattern in error_patterns:
                if pattern.error_type in misconception.indicator_patterns:
                    # More frequent = higher confidence
                    pattern_confidence = min(1.0, pattern.frequency / 10)
                    confidence = max(confidence, pattern_confidence)

            # Check for overconfident wrong answers
            wrong_responses = [r for r in responses if not r.is_correct]
            if wrong_responses:
                overconfident = [
                    r for r in wrong_responses
                    if r.confidence and r.confidence > 0.7
                ]
                if len(overconfident) >= 2:
                    confidence = max(confidence, 0.6)

            if confidence > 0.3:
                detected.append((misconception.misconception_id, round(confidence, 3)))

        # Sort by confidence
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected

    def _generate_remediations(
        self,
        detected_misconceptions: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """Generate remediation recommendations"""
        remediations = []

        for m_id, confidence in detected_misconceptions:
            misconception = self.library.get_misconception(m_id)
            if not misconception:
                continue

            for strategy in misconception.remediation_strategies[:2]:
                remediations.append({
                    "misconception_id": m_id,
                    "misconception_name": misconception.name,
                    "strategy": strategy.value,
                    "confidence": confidence,
                    "priority": self._get_remediation_priority(
                        misconception.severity, confidence
                    ),
                    "description": self._get_strategy_description(strategy),
                })

        # Sort by priority
        remediations.sort(key=lambda x: x["priority"], reverse=True)
        return remediations

    def _get_remediation_priority(
        self,
        severity: MisconceptionSeverity,
        confidence: float
    ) -> float:
        """Calculate remediation priority"""
        severity_weight = {
            MisconceptionSeverity.CRITICAL: 1.0,
            MisconceptionSeverity.HIGH: 0.8,
            MisconceptionSeverity.MEDIUM: 0.5,
            MisconceptionSeverity.LOW: 0.3,
        }
        return severity_weight[severity] * confidence

    def _get_strategy_description(self, strategy: RemediationStrategy) -> str:
        """Get human-readable description of remediation strategy"""
        descriptions = {
            RemediationStrategy.DIRECT_INSTRUCTION:
                "Provide clear, explicit explanation of the correct concept",
            RemediationStrategy.COGNITIVE_CONFLICT:
                "Present scenarios that reveal the misconception's incorrectness",
            RemediationStrategy.ANALOGICAL_BRIDGING:
                "Use familiar concepts as analogies to explain",
            RemediationStrategy.WORKED_EXAMPLES:
                "Show step-by-step examples of correct approach",
            RemediationStrategy.REFUTATIONAL_TEXT:
                "Directly address and refute the misconception",
            RemediationStrategy.SOCRATIC_DIALOGUE:
                "Guide understanding through strategic questioning",
            RemediationStrategy.VISUALIZATION:
                "Use visual representations to clarify the concept",
        }
        return descriptions.get(strategy, "Apply targeted remediation")


class MisconceptionTracker:
    """
    High-level tracker for misconception management

    Provides easy-to-use interface for the adaptive learning system
    """

    def __init__(self):
        self.detector = MentalModelDetector()
        self.response_history: Dict[int, List[ResponseData]] = {}

    def record_response(
        self,
        user_id: int,
        question_id: int,
        concept_id: int,
        is_correct: bool,
        selected_answer: str,
        correct_answer: str,
        response_time_ms: int,
        confidence: float = None,
        distractor_type: str = None,
    ) -> Dict[str, Any]:
        """
        Record a response and analyze for misconceptions

        Returns immediate analysis results
        """
        response = ResponseData(
            question_id=question_id,
            concept_id=concept_id,
            is_correct=is_correct,
            selected_answer=selected_answer,
            correct_answer=correct_answer,
            response_time_ms=response_time_ms,
            confidence=confidence,
            distractor_type=distractor_type,
        )

        # Store in history
        if user_id not in self.response_history:
            self.response_history[user_id] = []
        self.response_history[user_id].append(response)

        # Analyze
        analysis = self.detector.analyze_response(user_id, response)

        return {
            "response_recorded": True,
            "analysis": analysis,
            "mental_model": self._get_model_summary(user_id, concept_id),
        }

    def get_concept_diagnosis(
        self,
        user_id: int,
        concept_id: int,
    ) -> Dict[str, Any]:
        """Get comprehensive diagnosis for a concept"""
        responses = self.response_history.get(user_id, [])
        diagnosis = self.detector.diagnose_concept(user_id, concept_id, responses)

        return {
            "user_id": diagnosis.user_id,
            "concept_id": diagnosis.concept_id,
            "mental_model_accuracy": round(diagnosis.mental_model_accuracy, 3),
            "confidence_calibration": round(diagnosis.confidence_calibration, 3),
            "misconceptions": [
                {"id": m_id, "confidence": conf}
                for m_id, conf in diagnosis.detected_misconceptions
            ],
            "error_patterns": [
                {
                    "type": p.error_type,
                    "frequency": p.frequency,
                }
                for p in diagnosis.error_patterns_found
            ],
            "remediations": diagnosis.recommended_remediations[:3],
        }

    def get_user_summary(self, user_id: int) -> Dict[str, Any]:
        """Get misconception summary for user"""
        return self.detector.get_user_misconception_summary(user_id)

    def _get_model_summary(self, user_id: int, concept_id: int) -> Dict[str, Any]:
        """Get mental model summary"""
        model = self.detector.get_mental_model(user_id, concept_id)
        return {
            "accuracy": round(model.accuracy_score, 3),
            "completeness": round(model.completeness_score, 3),
            "stability": round(model.stability_score, 3),
            "assessment_count": model.assessment_count,
        }
