"""
Evidence-Centered Design (ECD) Framework

Implements the complete ECD assessment framework:
1. Competency Model - What we're measuring (KSAs)
2. Task Model - Tasks that elicit evidence
3. Evidence Model - How to interpret evidence
4. Assembly Model - How to combine evidence for claims

References:
- Mislevy et al., 2003: Evidence-Centered Assessment Design
- Shute, 2011: Stealth Assessment in Digital Games
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
import math
from pydantic import BaseModel

from .telemetry_collector import TelemetryEvent, TelemetryEventType, EvidenceRule


# ============================================================================
# COMPETENCY MODEL - What we're measuring
# ============================================================================

class CompetencyLevel(str, Enum):
    """Levels of competency mastery"""
    NOVICE = "novice"          # Limited understanding, needs guidance
    DEVELOPING = "developing"   # Partial understanding, some errors
    COMPETENT = "competent"     # Solid understanding, occasional errors
    PROFICIENT = "proficient"   # Strong understanding, rare errors
    EXPERT = "expert"           # Deep understanding, can teach others


class KnowledgeType(str, Enum):
    """Types of knowledge (Bloom's taxonomy aligned)"""
    DECLARATIVE = "declarative"     # Facts, concepts, terminology
    PROCEDURAL = "procedural"       # How to do something
    CONCEPTUAL = "conceptual"       # Understanding relationships
    METACOGNITIVE = "metacognitive" # Self-awareness, learning strategies


@dataclass
class Competency:
    """
    A competency being measured (Knowledge, Skill, or Ability)
    """
    id: str
    name: str
    description: str
    knowledge_type: KnowledgeType
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[int] = field(default_factory=list)

    # Thresholds for competency levels
    level_thresholds: Dict[CompetencyLevel, float] = field(default_factory=lambda: {
        CompetencyLevel.NOVICE: 0.0,
        CompetencyLevel.DEVELOPING: 0.25,
        CompetencyLevel.COMPETENT: 0.50,
        CompetencyLevel.PROFICIENT: 0.75,
        CompetencyLevel.EXPERT: 0.90,
    })

    def get_level(self, mastery: float) -> CompetencyLevel:
        """Get competency level based on mastery score"""
        for level in reversed(CompetencyLevel):
            if mastery >= self.level_thresholds[level]:
                return level
        return CompetencyLevel.NOVICE


@dataclass
class CompetencyModel:
    """
    Defines the competencies being assessed
    """
    competencies: Dict[str, Competency] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)  # prereq -> dependents

    def add_competency(self, competency: Competency):
        """Add a competency to the model"""
        self.competencies[competency.id] = competency

        # Update dependency graph
        for prereq_id in competency.prerequisites:
            if prereq_id not in self.dependency_graph:
                self.dependency_graph[prereq_id] = set()
            self.dependency_graph[prereq_id].add(competency.id)

    def get_prerequisite_chain(self, competency_id: str) -> List[str]:
        """Get ordered list of prerequisites (topological sort)"""
        if competency_id not in self.competencies:
            return []

        visited = set()
        chain = []

        def dfs(cid: str):
            if cid in visited:
                return
            visited.add(cid)
            comp = self.competencies.get(cid)
            if comp:
                for prereq in comp.prerequisites:
                    dfs(prereq)
                chain.append(cid)

        dfs(competency_id)
        return chain[:-1]  # Exclude the competency itself

    def get_dependents(self, competency_id: str) -> Set[str]:
        """Get competencies that depend on this one"""
        return self.dependency_graph.get(competency_id, set())


# ============================================================================
# TASK MODEL - What tasks elicit evidence
# ============================================================================

class TaskDifficulty(str, Enum):
    """Task difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class TaskType(str, Enum):
    """Types of tasks that elicit evidence"""
    CONTENT_CONSUMPTION = "content_consumption"  # Reading, watching
    EXPLORATION = "exploration"                  # Navigating, searching
    PRACTICE = "practice"                        # Solving problems
    APPLICATION = "application"                  # Applying knowledge
    CREATION = "creation"                        # Creating artifacts
    EXPLANATION = "explanation"                  # Teaching, explaining
    TRANSFER = "transfer"                        # Novel contexts


@dataclass
class TaskFeatures:
    """
    Features of a task that affect evidence interpretation
    """
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    time_limit: Optional[int] = None  # Seconds
    scaffolding_level: float = 0.0    # 0 = no help, 1 = full guidance
    novelty: float = 0.5              # 0 = familiar, 1 = completely new
    complexity: float = 0.5           # 0 = simple, 1 = complex
    interactivity: float = 0.5        # 0 = passive, 1 = highly interactive


@dataclass
class TaskModel:
    """
    Defines a task that elicits evidence for competencies
    """
    id: str
    name: str
    task_type: TaskType
    features: TaskFeatures
    target_competencies: List[str]  # Competency IDs this task measures
    evidence_weights: Dict[str, float] = field(default_factory=dict)  # competency_id -> weight

    # Observable behaviors that constitute evidence
    observable_behaviors: List[str] = field(default_factory=list)

    # Task-specific scoring rubric
    scoring_rubric: Dict[str, Any] = field(default_factory=dict)

    def get_evidence_weight(self, competency_id: str) -> float:
        """Get weight for this task's evidence toward a competency"""
        if competency_id in self.evidence_weights:
            return self.evidence_weights[competency_id]
        if competency_id in self.target_competencies:
            return 1.0
        return 0.0


class TaskModelRegistry:
    """
    Registry of task models for evidence collection
    """

    def __init__(self):
        self.tasks: Dict[str, TaskModel] = {}
        self.tasks_by_competency: Dict[str, List[str]] = {}  # competency -> task_ids

        # Register default task models
        self._register_default_tasks()

    def _register_default_tasks(self):
        """Register default stealth assessment task models"""

        # Content reading task
        self.register(TaskModel(
            id="content_reading",
            name="Content Reading",
            task_type=TaskType.CONTENT_CONSUMPTION,
            features=TaskFeatures(difficulty=TaskDifficulty.EASY),
            target_competencies=["declarative_knowledge"],
            observable_behaviors=[
                "dwell_time_pattern",
                "scroll_depth",
                "section_revisits",
                "highlight_selections",
            ],
            scoring_rubric={
                "optimal_dwell_ratio": (0.8, 1.5),
                "scroll_completion_threshold": 0.8,
                "revisit_indicates_mastery": False,
            }
        ))

        # Video learning task
        self.register(TaskModel(
            id="video_learning",
            name="Video Learning",
            task_type=TaskType.CONTENT_CONSUMPTION,
            features=TaskFeatures(difficulty=TaskDifficulty.EASY, interactivity=0.3),
            target_competencies=["declarative_knowledge", "procedural_knowledge"],
            observable_behaviors=[
                "watch_completion_rate",
                "replay_frequency",
                "pause_for_notes",
                "playback_speed",
            ],
            scoring_rubric={
                "completion_threshold": 0.9,
                "replay_indicates": "careful_learning",
                "speed_1.5x_indicates": "prior_knowledge",
            }
        ))

        # Interactive exploration task
        self.register(TaskModel(
            id="concept_exploration",
            name="Concept Exploration",
            task_type=TaskType.EXPLORATION,
            features=TaskFeatures(
                difficulty=TaskDifficulty.MEDIUM,
                interactivity=0.8
            ),
            target_competencies=["conceptual_knowledge", "metacognitive"],
            observable_behaviors=[
                "navigation_pattern",
                "related_concept_visits",
                "search_query_sophistication",
                "time_on_difficult_concepts",
            ],
            evidence_weights={
                "conceptual_knowledge": 0.8,
                "metacognitive": 0.5,
            }
        ))

        # Chat-based inquiry task
        self.register(TaskModel(
            id="chat_inquiry",
            name="Chat-Based Inquiry",
            task_type=TaskType.EXPLORATION,
            features=TaskFeatures(
                difficulty=TaskDifficulty.MEDIUM,
                interactivity=1.0
            ),
            target_competencies=["conceptual_knowledge", "metacognitive"],
            observable_behaviors=[
                "question_depth",
                "follow_up_questions",
                "concept_terminology_use",
                "question_complexity_progression",
            ],
            scoring_rubric={
                "surface_question_keywords": ["what", "define", "list"],
                "deep_question_keywords": ["why", "how", "compare", "analyze"],
                "synthesis_keywords": ["relate", "apply", "evaluate"],
            }
        ))

        # Practice problem task
        self.register(TaskModel(
            id="practice_problem",
            name="Practice Problem",
            task_type=TaskType.PRACTICE,
            features=TaskFeatures(
                difficulty=TaskDifficulty.MEDIUM,
                interactivity=0.7
            ),
            target_competencies=["procedural_knowledge", "application"],
            observable_behaviors=[
                "time_to_first_attempt",
                "attempt_count",
                "hint_usage",
                "error_pattern",
                "self_correction",
            ],
            evidence_weights={
                "procedural_knowledge": 1.0,
                "application": 0.7,
            }
        ))

        # Knowledge transfer task
        self.register(TaskModel(
            id="knowledge_transfer",
            name="Knowledge Transfer",
            task_type=TaskType.TRANSFER,
            features=TaskFeatures(
                difficulty=TaskDifficulty.HARD,
                novelty=0.8
            ),
            target_competencies=["conceptual_knowledge", "application", "transfer"],
            observable_behaviors=[
                "successful_novel_application",
                "analogy_recognition",
                "cross_domain_connection",
            ]
        ))

    def register(self, task: TaskModel):
        """Register a task model"""
        self.tasks[task.id] = task

        for comp_id in task.target_competencies:
            if comp_id not in self.tasks_by_competency:
                self.tasks_by_competency[comp_id] = []
            self.tasks_by_competency[comp_id].append(task.id)

    def get_tasks_for_competency(self, competency_id: str) -> List[TaskModel]:
        """Get all tasks that measure a competency"""
        task_ids = self.tasks_by_competency.get(competency_id, [])
        return [self.tasks[tid] for tid in task_ids if tid in self.tasks]

    def identify_task(self, events: List[TelemetryEvent]) -> Optional[TaskModel]:
        """Identify the task being performed from telemetry events"""
        if not events:
            return None

        event_types = set(e.event_type for e in events)

        # Video learning
        if TelemetryEventType.VIDEO_PLAY in event_types:
            return self.tasks.get("video_learning")

        # Chat inquiry
        if TelemetryEventType.CHAT_QUERY in event_types:
            return self.tasks.get("chat_inquiry")

        # Content reading
        if TelemetryEventType.CONTENT_DWELL in event_types:
            return self.tasks.get("content_reading")

        # Concept exploration
        if TelemetryEventType.CONCEPT_CLICK in event_types:
            return self.tasks.get("concept_exploration")

        # Practice problem
        if TelemetryEventType.QUIZ_ATTEMPT in event_types:
            return self.tasks.get("practice_problem")

        return None


# ============================================================================
# EVIDENCE MODEL - How to interpret evidence
# ============================================================================

@dataclass
class EvidenceObservation:
    """
    A single piece of evidence from an observable behavior
    """
    task_id: str
    competency_id: str
    timestamp: datetime
    raw_value: Any                    # Raw observed value
    normalized_score: float           # 0-1 normalized score
    confidence: float                 # How reliable is this evidence (0-1)
    evidence_type: str               # Type of evidence
    task_context: Dict[str, Any] = field(default_factory=dict)

    def weighted_score(self) -> float:
        """Score weighted by confidence"""
        return self.normalized_score * self.confidence


class EvidenceAccumulator(str, Enum):
    """Methods for accumulating evidence over time"""
    LATEST = "latest"           # Use most recent evidence only
    AVERAGE = "average"         # Simple average
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by recency/confidence
    BAYESIAN = "bayesian"       # Bayesian update
    MAX = "max"                 # Maximum score (benefit of doubt)
    TREND = "trend"             # Consider improvement trend


@dataclass
class EvidenceRule_ECD(EvidenceRule):
    """
    Enhanced evidence rule for ECD framework
    """
    target_behaviors: List[str] = field(default_factory=list)
    reliability: float = 0.8  # How reliable is this evidence type
    minimum_observations: int = 1
    decay_rate: float = 0.95  # Evidence decay over time

    def calculate_confidence(
        self,
        num_observations: int,
        time_since_observation: timedelta
    ) -> float:
        """Calculate confidence based on observations and recency"""
        # More observations = higher confidence
        obs_confidence = min(1.0, num_observations / 5)

        # Recent observations = higher confidence
        hours_elapsed = time_since_observation.total_seconds() / 3600
        recency_confidence = self.decay_rate ** (hours_elapsed / 24)

        return self.reliability * obs_confidence * recency_confidence


class ContentEngagementEvidenceRule(EvidenceRule_ECD):
    """
    Evidence from content engagement patterns
    """

    def __init__(self):
        super().__init__(
            name="content_engagement",
            weight=0.8,
            reliability=0.75
        )
        self.target_behaviors = [
            "dwell_time_pattern",
            "scroll_depth",
            "section_revisits",
        ]

    def evaluate(self, events: List[TelemetryEvent]) -> Optional[float]:
        """Evaluate content engagement evidence"""
        dwell_events = [
            e for e in events
            if e.event_type in [
                TelemetryEventType.PAGE_VIEW,
                TelemetryEventType.CONTENT_DWELL
            ]
        ]

        if not dwell_events:
            return None

        scores = []

        # Dwell time analysis
        total_dwell = sum(
            e.data.get("duration_seconds", 0) for e in dwell_events
        )
        word_count = dwell_events[0].data.get("word_count", 500)
        expected_time = (word_count / 250) * 60

        if expected_time > 0:
            ratio = total_dwell / expected_time
            if 0.8 <= ratio <= 1.5:
                scores.append(0.9)  # Optimal
            elif 0.5 <= ratio < 0.8 or 1.5 < ratio <= 2.5:
                scores.append(0.7)  # Acceptable
            elif ratio < 0.5:
                scores.append(0.3)  # Too fast
            else:
                scores.append(0.5)  # Too slow

        # Scroll depth analysis
        max_scroll = max(
            (e.data.get("scroll_depth", 0) for e in dwell_events),
            default=0
        )
        scores.append(min(1.0, max_scroll))

        # Section revisit analysis
        section_visits = {}
        for e in dwell_events:
            section = e.data.get("section_id", "main")
            section_visits[section] = section_visits.get(section, 0) + 1

        revisited_sections = sum(1 for v in section_visits.values() if v > 1)
        if len(section_visits) > 0:
            revisit_ratio = revisited_sections / len(section_visits)
            # Some revisits indicate careful reading
            if revisit_ratio > 0.5:
                scores.append(0.8)
            elif revisit_ratio > 0.2:
                scores.append(0.9)
            else:
                scores.append(0.7)

        return sum(scores) / len(scores) if scores else None


class QuerySophisticationEvidenceRule(EvidenceRule_ECD):
    """
    Evidence from query sophistication in chat interactions
    """

    def __init__(self):
        super().__init__(
            name="query_sophistication",
            weight=0.85,
            reliability=0.8
        )
        self.target_behaviors = [
            "question_depth",
            "follow_up_questions",
            "concept_terminology_use",
        ]

        # Keywords indicating question depth (Bloom's taxonomy aligned)
        self.depth_indicators = {
            "remember": ["what", "when", "who", "where", "define", "list", "name"],
            "understand": ["explain", "describe", "summarize", "compare", "contrast"],
            "apply": ["how", "use", "implement", "solve", "demonstrate"],
            "analyze": ["why", "analyze", "examine", "differentiate", "relationship"],
            "evaluate": ["evaluate", "judge", "critique", "assess", "recommend"],
            "create": ["design", "create", "propose", "develop", "formulate"],
        }

        self.depth_scores = {
            "remember": 0.3,
            "understand": 0.5,
            "apply": 0.7,
            "analyze": 0.85,
            "evaluate": 0.95,
            "create": 1.0,
        }

    def evaluate(self, events: List[TelemetryEvent]) -> Optional[float]:
        """Evaluate query sophistication"""
        chat_events = [
            e for e in events
            if e.event_type == TelemetryEventType.CHAT_QUERY
        ]

        if not chat_events:
            return None

        query_scores = []

        for event in chat_events:
            query = event.data.get("query", "").lower()

            # Determine depth level
            max_depth = "remember"
            max_depth_score = 0.3

            for level, keywords in self.depth_indicators.items():
                for keyword in keywords:
                    if keyword in query:
                        if self.depth_scores[level] > max_depth_score:
                            max_depth = level
                            max_depth_score = self.depth_scores[level]

            # Bonus for longer, more detailed questions
            length_bonus = min(0.1, len(query) / 500)

            query_scores.append(min(1.0, max_depth_score + length_bonus))

        if not query_scores:
            return None

        # Check for progression (asking deeper questions over time)
        if len(query_scores) >= 3:
            # Trend analysis
            first_half = sum(query_scores[:len(query_scores)//2]) / (len(query_scores)//2)
            second_half = sum(query_scores[len(query_scores)//2:]) / (len(query_scores) - len(query_scores)//2)

            if second_half > first_half:
                # Improving - bonus
                return min(1.0, sum(query_scores) / len(query_scores) + 0.1)

        return sum(query_scores) / len(query_scores)


class ProblemSolvingEvidenceRule(EvidenceRule_ECD):
    """
    Evidence from problem-solving behavior
    """

    def __init__(self):
        super().__init__(
            name="problem_solving",
            weight=0.95,
            reliability=0.9
        )
        self.target_behaviors = [
            "time_to_first_attempt",
            "attempt_count",
            "hint_usage",
            "self_correction",
        ]

    def evaluate(self, events: List[TelemetryEvent]) -> Optional[float]:
        """Evaluate problem-solving evidence"""
        quiz_events = [
            e for e in events
            if e.event_type == TelemetryEventType.QUIZ_ATTEMPT
        ]

        if not quiz_events:
            return None

        scores = []

        for event in quiz_events:
            data = event.data

            # Correctness (primary signal)
            correct = data.get("correct", False)
            base_score = 0.8 if correct else 0.2

            # Time to attempt (normalized by expected time)
            time_taken = data.get("time_seconds", 0)
            expected_time = data.get("expected_time", 60)
            if expected_time > 0 and time_taken > 0:
                time_ratio = time_taken / expected_time
                if 0.3 <= time_ratio <= 1.5:
                    time_modifier = 0.1  # Good pace
                elif time_ratio < 0.3:
                    time_modifier = 0.05 if correct else -0.1  # Too fast
                else:
                    time_modifier = -0.05  # Struggled
            else:
                time_modifier = 0

            # Hint usage (reduces score)
            hints_used = data.get("hints_used", 0)
            hint_penalty = hints_used * 0.1

            # Self-correction (positive signal)
            self_corrected = data.get("self_corrected", False)
            correction_bonus = 0.1 if self_corrected else 0

            # Attempt count (fewer is better for correct, more shows persistence for incorrect)
            attempt_count = data.get("attempt_count", 1)
            if correct:
                attempt_modifier = max(-0.2, -(attempt_count - 1) * 0.1)
            else:
                attempt_modifier = min(0.1, attempt_count * 0.02)  # Persistence bonus

            final_score = base_score + time_modifier - hint_penalty + correction_bonus + attempt_modifier
            scores.append(max(0.0, min(1.0, final_score)))

        return sum(scores) / len(scores) if scores else None


# ============================================================================
# ASSEMBLY MODEL - How to combine evidence for claims
# ============================================================================

@dataclass
class CompetencyClaim:
    """
    A claim about a student's competency level
    """
    competency_id: str
    level: CompetencyLevel
    probability: float
    confidence: float
    evidence_count: int
    last_updated: datetime
    trend: str = "stable"  # "improving", "stable", "declining"


class AssemblyModel:
    """
    Assembles evidence into claims about competencies

    Implements multiple evidence combination strategies:
    - Weighted averaging
    - Bayesian updating
    - Trend analysis
    """

    def __init__(
        self,
        accumulator: EvidenceAccumulator = EvidenceAccumulator.WEIGHTED_AVERAGE,
        prior: float = 0.2,  # Prior probability for Bayesian
        decay_hours: float = 168,  # Evidence decay half-life (1 week)
    ):
        self.accumulator = accumulator
        self.prior = prior
        self.decay_hours = decay_hours

        # Evidence storage per user-competency
        self.evidence_store: Dict[str, List[EvidenceObservation]] = {}

    def _get_key(self, user_id: str, competency_id: str) -> str:
        return f"{user_id}:{competency_id}"

    def add_evidence(
        self,
        user_id: str,
        observation: EvidenceObservation
    ):
        """Add evidence observation"""
        key = self._get_key(user_id, observation.competency_id)

        if key not in self.evidence_store:
            self.evidence_store[key] = []

        self.evidence_store[key].append(observation)

        # Limit evidence history
        if len(self.evidence_store[key]) > 100:
            self.evidence_store[key] = self.evidence_store[key][-100:]

    def get_evidence(
        self,
        user_id: str,
        competency_id: str,
        since: Optional[datetime] = None
    ) -> List[EvidenceObservation]:
        """Get evidence for a user-competency pair"""
        key = self._get_key(user_id, competency_id)
        evidence = self.evidence_store.get(key, [])

        if since:
            evidence = [e for e in evidence if e.timestamp >= since]

        return evidence

    def assemble_claim(
        self,
        user_id: str,
        competency_id: str,
        competency: Optional[Competency] = None
    ) -> CompetencyClaim:
        """
        Assemble evidence into a claim about competency
        """
        evidence = self.get_evidence(user_id, competency_id)

        if not evidence:
            return CompetencyClaim(
                competency_id=competency_id,
                level=CompetencyLevel.NOVICE,
                probability=self.prior,
                confidence=0.0,
                evidence_count=0,
                last_updated=datetime.utcnow(),
            )

        now = datetime.utcnow()

        # Calculate probability based on accumulation method
        if self.accumulator == EvidenceAccumulator.LATEST:
            probability = evidence[-1].normalized_score

        elif self.accumulator == EvidenceAccumulator.AVERAGE:
            probability = sum(e.normalized_score for e in evidence) / len(evidence)

        elif self.accumulator == EvidenceAccumulator.WEIGHTED_AVERAGE:
            probability = self._weighted_average(evidence, now)

        elif self.accumulator == EvidenceAccumulator.BAYESIAN:
            probability = self._bayesian_update(evidence)

        elif self.accumulator == EvidenceAccumulator.MAX:
            probability = max(e.normalized_score for e in evidence)

        elif self.accumulator == EvidenceAccumulator.TREND:
            probability = self._trend_based(evidence, now)

        else:
            probability = sum(e.normalized_score for e in evidence) / len(evidence)

        # Calculate confidence
        confidence = self._calculate_confidence(evidence, now)

        # Determine trend
        trend = self._analyze_trend(evidence)

        # Determine level
        if competency:
            level = competency.get_level(probability)
        else:
            level = self._default_level(probability)

        return CompetencyClaim(
            competency_id=competency_id,
            level=level,
            probability=probability,
            confidence=confidence,
            evidence_count=len(evidence),
            last_updated=now,
            trend=trend,
        )

    def _weighted_average(
        self,
        evidence: List[EvidenceObservation],
        now: datetime
    ) -> float:
        """Calculate weighted average with recency and confidence weighting"""
        if not evidence:
            return self.prior

        total_weight = 0.0
        weighted_sum = 0.0

        for obs in evidence:
            # Recency weight (exponential decay)
            hours_ago = (now - obs.timestamp).total_seconds() / 3600
            recency_weight = math.exp(-hours_ago / self.decay_hours)

            # Combined weight
            weight = recency_weight * obs.confidence
            weighted_sum += obs.normalized_score * weight
            total_weight += weight

        if total_weight == 0:
            return self.prior

        return weighted_sum / total_weight

    def _bayesian_update(self, evidence: List[EvidenceObservation]) -> float:
        """Bayesian update of probability"""
        # Start with prior
        probability = self.prior

        for obs in evidence:
            # Simple Bayesian update
            # P(competent | evidence) = P(evidence | competent) * P(competent) / P(evidence)

            # Likelihood ratio based on evidence score
            if obs.normalized_score > 0.5:
                # Positive evidence
                likelihood_ratio = obs.normalized_score / (1 - obs.normalized_score + 0.01)
            else:
                # Negative evidence
                likelihood_ratio = obs.normalized_score / (1 - obs.normalized_score + 0.01)

            # Update with confidence weighting
            effective_lr = 1 + (likelihood_ratio - 1) * obs.confidence

            # Bayes update
            odds = probability / (1 - probability + 0.001)
            new_odds = odds * effective_lr
            probability = new_odds / (1 + new_odds)

        return max(0.01, min(0.99, probability))

    def _trend_based(
        self,
        evidence: List[EvidenceObservation],
        now: datetime
    ) -> float:
        """Probability with trend consideration"""
        base = self._weighted_average(evidence, now)

        if len(evidence) < 3:
            return base

        # Calculate recent trend
        recent = evidence[-5:]  # Last 5 observations
        earlier = evidence[:-5][-5:] if len(evidence) > 5 else []

        if not earlier:
            return base

        recent_avg = sum(e.normalized_score for e in recent) / len(recent)
        earlier_avg = sum(e.normalized_score for e in earlier) / len(earlier)

        trend = recent_avg - earlier_avg

        # Adjust probability based on trend
        # Positive trend: boost probability
        # Negative trend: reduce probability
        adjustment = trend * 0.2  # 20% of trend

        return max(0.01, min(0.99, base + adjustment))

    def _calculate_confidence(
        self,
        evidence: List[EvidenceObservation],
        now: datetime
    ) -> float:
        """Calculate confidence in the claim"""
        if not evidence:
            return 0.0

        # Factors:
        # 1. Number of observations
        count_factor = min(1.0, len(evidence) / 10)

        # 2. Recency of evidence
        most_recent = max(e.timestamp for e in evidence)
        hours_since = (now - most_recent).total_seconds() / 3600
        recency_factor = math.exp(-hours_since / self.decay_hours)

        # 3. Consistency of evidence
        scores = [e.normalized_score for e in evidence]
        if len(scores) > 1:
            variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            consistency_factor = 1 - min(1.0, variance * 4)  # Higher variance = lower confidence
        else:
            consistency_factor = 0.5

        # 4. Average evidence confidence
        avg_confidence = sum(e.confidence for e in evidence) / len(evidence)

        return (
            count_factor * 0.25 +
            recency_factor * 0.25 +
            consistency_factor * 0.25 +
            avg_confidence * 0.25
        )

    def _analyze_trend(self, evidence: List[EvidenceObservation]) -> str:
        """Analyze trend in evidence"""
        if len(evidence) < 3:
            return "stable"

        # Compare recent vs older
        n = len(evidence)
        first_third = evidence[:n//3]
        last_third = evidence[-(n//3):]

        if not first_third or not last_third:
            return "stable"

        first_avg = sum(e.normalized_score for e in first_third) / len(first_third)
        last_avg = sum(e.normalized_score for e in last_third) / len(last_third)

        diff = last_avg - first_avg

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"

    def _default_level(self, probability: float) -> CompetencyLevel:
        """Default level mapping"""
        if probability >= 0.90:
            return CompetencyLevel.EXPERT
        elif probability >= 0.75:
            return CompetencyLevel.PROFICIENT
        elif probability >= 0.50:
            return CompetencyLevel.COMPETENT
        elif probability >= 0.25:
            return CompetencyLevel.DEVELOPING
        else:
            return CompetencyLevel.NOVICE


# ============================================================================
# INTEGRATED ECD ASSESSOR
# ============================================================================

class ECDAssessor:
    """
    Complete ECD-based stealth assessor

    Integrates:
    - Competency Model (what we measure)
    - Task Model (how we elicit evidence)
    - Evidence Model (how we interpret)
    - Assembly Model (how we combine)
    """

    def __init__(
        self,
        accumulator: EvidenceAccumulator = EvidenceAccumulator.WEIGHTED_AVERAGE
    ):
        self.competency_model = CompetencyModel()
        self.task_registry = TaskModelRegistry()
        self.assembly_model = AssemblyModel(accumulator=accumulator)

        # Evidence rules
        self.evidence_rules: List[EvidenceRule_ECD] = [
            ContentEngagementEvidenceRule(),
            QuerySophisticationEvidenceRule(),
            ProblemSolvingEvidenceRule(),
        ]

        # Concept to competency mapping
        self.concept_competency_map: Dict[int, List[str]] = {}

    def register_competency(self, competency: Competency):
        """Register a competency"""
        self.competency_model.add_competency(competency)

        # Map related concepts
        for concept_id in competency.related_concepts:
            if concept_id not in self.concept_competency_map:
                self.concept_competency_map[concept_id] = []
            self.concept_competency_map[concept_id].append(competency.id)

    def map_concept_to_competencies(
        self,
        concept_id: int,
        competency_ids: List[str]
    ):
        """Map a concept to competencies"""
        self.concept_competency_map[concept_id] = competency_ids

    def process_events(
        self,
        user_id: str,
        events: List[TelemetryEvent]
    ) -> Dict[str, CompetencyClaim]:
        """
        Process telemetry events and update competency claims

        Args:
            user_id: User identifier
            events: Telemetry events to process

        Returns:
            Updated claims for affected competencies
        """
        if not events:
            return {}

        # Identify the task being performed
        task = self.task_registry.identify_task(events)

        # Get affected competencies from task or concept
        affected_competencies: Set[str] = set()

        if task:
            affected_competencies.update(task.target_competencies)

        # Also check concept mapping
        for event in events:
            if event.concept_id:
                comp_ids = self.concept_competency_map.get(event.concept_id, [])
                affected_competencies.update(comp_ids)

        # Apply evidence rules
        for rule in self.evidence_rules:
            score = rule.evaluate(events)

            if score is not None:
                # Create evidence observations for each affected competency
                for comp_id in affected_competencies:
                    # Get weight from task model if available
                    weight = 1.0
                    if task:
                        weight = task.get_evidence_weight(comp_id)

                    observation = EvidenceObservation(
                        task_id=task.id if task else "unknown",
                        competency_id=comp_id,
                        timestamp=events[0].timestamp if events else datetime.utcnow(),
                        raw_value=score,
                        normalized_score=score * weight,
                        confidence=rule.reliability,
                        evidence_type=rule.name,
                        task_context={
                            "event_count": len(events),
                            "task_difficulty": task.features.difficulty.value if task else "unknown",
                        }
                    )

                    self.assembly_model.add_evidence(user_id, observation)

        # Assemble claims
        claims = {}
        for comp_id in affected_competencies:
            competency = self.competency_model.competencies.get(comp_id)
            claims[comp_id] = self.assembly_model.assemble_claim(
                user_id, comp_id, competency
            )

        return claims

    def get_competency_claim(
        self,
        user_id: str,
        competency_id: str
    ) -> CompetencyClaim:
        """Get current claim for a competency"""
        competency = self.competency_model.competencies.get(competency_id)
        return self.assembly_model.assemble_claim(user_id, competency_id, competency)

    def get_all_claims(self, user_id: str) -> Dict[str, CompetencyClaim]:
        """Get all competency claims for a user"""
        claims = {}
        for comp_id in self.competency_model.competencies:
            claims[comp_id] = self.get_competency_claim(user_id, comp_id)
        return claims

    def get_learning_recommendations(
        self,
        user_id: str,
        target_competency: Optional[str] = None
    ) -> List[Dict]:
        """
        Get learning recommendations based on competency claims

        Returns tasks/content to address competency gaps
        """
        claims = self.get_all_claims(user_id)
        recommendations = []

        for comp_id, claim in claims.items():
            if target_competency and comp_id != target_competency:
                continue

            if claim.level in [CompetencyLevel.NOVICE, CompetencyLevel.DEVELOPING]:
                # Find tasks that can help
                tasks = self.task_registry.get_tasks_for_competency(comp_id)

                # Prioritize by difficulty (easier first for lower levels)
                tasks.sort(key=lambda t: (
                    0 if t.features.difficulty == TaskDifficulty.EASY else
                    1 if t.features.difficulty == TaskDifficulty.MEDIUM else 2
                ))

                for task in tasks[:2]:  # Top 2 tasks
                    recommendations.append({
                        "competency_id": comp_id,
                        "competency_level": claim.level.value,
                        "task_id": task.id,
                        "task_name": task.name,
                        "task_type": task.task_type.value,
                        "priority": "high" if claim.level == CompetencyLevel.NOVICE else "medium",
                        "confidence": claim.confidence,
                        "trend": claim.trend,
                    })

        return recommendations
