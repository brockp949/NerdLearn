"""
Intervention Engine - Adaptive Learning Support System

Research alignment:
- Just-in-Time Interventions: Deliver support when learner needs it most
- Scaffolding Theory (Vygotsky): Graduated support within ZPD
- Self-Determination Theory: Balance structure with autonomy
- Productive Failure: Sometimes struggle is beneficial

This engine orchestrates:
1. Frustration Detection → Emotional support interventions
2. Cognitive Load → Difficulty adjustments, scaffolding
3. Metacognition → Reflection prompts, calibration feedback
4. Error Patterns → Targeted remediation

Key Principle: Intervene enough to prevent disengagement,
but not so much that productive struggle is interrupted.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import random

from .frustration_detector import (
    FrustrationDetector,
    FrustrationLevel,
    StruggleType,
    FrustrationEstimate,
    InteractionEvent,
)
from .metacognition import (
    MetacognitionPrompter,
    MetacognitionPromptType,
    MetacognitionPrompt,
    CalibrationTracker,
    CalibrationLevel,
)

logger = logging.getLogger(__name__)


class InterventionType(str, Enum):
    """Types of learning interventions"""
    # Emotional/Motivational
    ENCOURAGEMENT = "encouragement"
    BREAK_SUGGESTION = "break_suggestion"
    PROGRESS_REMINDER = "progress_reminder"
    GROWTH_MINDSET = "growth_mindset"

    # Cognitive Support
    HINT = "hint"
    WORKED_EXAMPLE = "worked_example"
    SIMPLIFY_CONTENT = "simplify_content"
    PREREQUISITE_REVIEW = "prerequisite_review"
    SCAFFOLD = "scaffold"

    # Metacognitive
    REFLECTION_PROMPT = "reflection_prompt"
    SELF_EXPLANATION = "self_explanation"
    STRATEGY_SUGGESTION = "strategy_suggestion"
    CALIBRATION_FEEDBACK = "calibration_feedback"

    # Content Adjustment
    REDUCE_DIFFICULTY = "reduce_difficulty"
    INCREASE_DIFFICULTY = "increase_difficulty"
    CHANGE_MODALITY = "change_modality"
    PRACTICE_BREAK = "practice_break"

    # None
    NONE = "none"


class InterventionPriority(str, Enum):
    """Priority levels for interventions"""
    CRITICAL = "critical"    # Must show immediately
    HIGH = "high"            # Show soon
    MEDIUM = "medium"        # Show when appropriate
    LOW = "low"              # Optional, can skip


@dataclass
class Intervention:
    """A learning intervention to deliver"""
    intervention_id: str
    intervention_type: InterventionType
    priority: InterventionPriority
    title: str
    message: str
    action: Optional[str] = None  # Button/action text
    action_data: Dict = field(default_factory=dict)
    display_duration_seconds: Optional[int] = None
    dismissible: bool = True
    follow_ups: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class LearnerState:
    """Current state of the learner for intervention decisions"""
    user_id: str
    frustration: FrustrationEstimate
    cognitive_load_score: float
    cognitive_load_level: str
    calibration_level: CalibrationLevel
    consecutive_errors: int
    time_on_task_minutes: float
    session_duration_minutes: float
    concepts_mastered_today: int
    last_intervention_time: Optional[datetime] = None
    last_intervention_type: Optional[InterventionType] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class InterventionDecision:
    """Decision about whether and how to intervene"""
    should_intervene: bool
    intervention: Optional[Intervention]
    reason: str
    alternative_interventions: List[Intervention] = field(default_factory=list)
    cooldown_seconds: int = 0


class InterventionEngine:
    """
    Orchestrates adaptive interventions based on learner state

    Design Principles:
    1. Don't over-intervene: Too many prompts = annoyance
    2. Respect productive struggle: Not all difficulty is bad
    3. Personalize: Different learners need different support
    4. Be timely: Right intervention at right time
    5. Fade support: Gradually remove scaffolding as learner improves
    """

    # Intervention cooldowns (minimum time between interventions)
    COOLDOWNS = {
        InterventionType.ENCOURAGEMENT: 180,       # 3 min
        InterventionType.BREAK_SUGGESTION: 600,    # 10 min
        InterventionType.HINT: 60,                 # 1 min
        InterventionType.REFLECTION_PROMPT: 300,   # 5 min
        InterventionType.SCAFFOLD: 120,            # 2 min
        InterventionType.CALIBRATION_FEEDBACK: 600, # 10 min
    }

    # Intervention messages
    MESSAGES = {
        InterventionType.ENCOURAGEMENT: [
            "You're making progress! Keep going.",
            "Learning takes time. You've got this!",
            "Every expert was once a beginner.",
            "Mistakes are part of learning. Keep trying!",
            "You're working hard. That's what matters.",
        ],
        InterventionType.BREAK_SUGGESTION: [
            "You've been working for a while. A short break might help!",
            "Consider taking a 5-minute break to refresh your mind.",
            "Your brain learns better with rest. Take a quick break?",
            "Stepping away briefly can help concepts sink in.",
        ],
        InterventionType.GROWTH_MINDSET: [
            "Remember: your brain grows stronger when you tackle hard problems.",
            "Struggling means you're learning. It's a good sign!",
            "Intelligence isn't fixed—it grows with effort.",
            "This is challenging because it's new. That's how learning works.",
        ],
        InterventionType.PROGRESS_REMINDER: [
            "Look how far you've come! You've mastered {mastered_count} concepts.",
            "You've been at this for {minutes} minutes and making real progress.",
            "Remember when this seemed impossible? You're doing it!",
        ],
        InterventionType.STRATEGY_SUGGESTION: [
            "Try explaining this concept out loud in your own words.",
            "Maybe try working through a simpler example first?",
            "Drawing a diagram might help you visualize this.",
            "Try breaking this problem into smaller steps.",
        ],
    }

    def __init__(
        self,
        frustration_detector: Optional[FrustrationDetector] = None,
        metacognition_prompter: Optional[MetacognitionPrompter] = None,
        intervention_frequency: float = 0.4,  # Max probability of intervening
        min_cooldown_seconds: int = 30,
    ):
        self.frustration_detector = frustration_detector or FrustrationDetector()
        self.metacognition_prompter = metacognition_prompter or MetacognitionPrompter()
        self.intervention_frequency = intervention_frequency
        self.min_cooldown = timedelta(seconds=min_cooldown_seconds)
        self.intervention_history: Dict[str, List[Intervention]] = {}

    def _check_cooldown(
        self,
        user_id: str,
        intervention_type: InterventionType,
        last_time: Optional[datetime]
    ) -> bool:
        """Check if intervention type is on cooldown"""
        if not last_time:
            return False

        cooldown = self.COOLDOWNS.get(intervention_type, 60)
        return datetime.utcnow() - last_time < timedelta(seconds=cooldown)

    def _get_random_message(self, intervention_type: InterventionType, context: Dict) -> str:
        """Get a random message for intervention type, with context substitution"""
        messages = self.MESSAGES.get(intervention_type, ["Keep going!"])
        message = random.choice(messages)

        # Substitute context variables
        return message.format(**context)

    def _create_frustration_intervention(
        self,
        frustration: FrustrationEstimate,
        context: Dict
    ) -> Optional[Intervention]:
        """Create intervention for frustration"""
        if frustration.level == FrustrationLevel.NONE:
            return None

        import uuid

        if frustration.level == FrustrationLevel.MILD:
            if frustration.struggle_type == StruggleType.PRODUCTIVE:
                # Let them continue, maybe light encouragement
                return Intervention(
                    intervention_id=str(uuid.uuid4())[:8],
                    intervention_type=InterventionType.ENCOURAGEMENT,
                    priority=InterventionPriority.LOW,
                    title="Keep Going!",
                    message=self._get_random_message(InterventionType.ENCOURAGEMENT, context),
                    dismissible=True,
                    display_duration_seconds=5,
                )
            return None

        if frustration.level == FrustrationLevel.MODERATE:
            if frustration.recommended_action == "slow_down_prompt":
                return Intervention(
                    intervention_id=str(uuid.uuid4())[:8],
                    intervention_type=InterventionType.STRATEGY_SUGGESTION,
                    priority=InterventionPriority.MEDIUM,
                    title="Take Your Time",
                    message="Slow down and read the question carefully. What is it really asking?",
                    action="Got it",
                    dismissible=True,
                )
            elif frustration.recommended_action == "offer_hint":
                return Intervention(
                    intervention_id=str(uuid.uuid4())[:8],
                    intervention_type=InterventionType.HINT,
                    priority=InterventionPriority.MEDIUM,
                    title="Need a Hint?",
                    message="You seem to be stuck. Would you like a hint to get started?",
                    action="Show Hint",
                    action_data={"show_hint": True},
                    dismissible=True,
                )

        if frustration.level == FrustrationLevel.HIGH:
            return Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                intervention_type=InterventionType.SCAFFOLD,
                priority=InterventionPriority.HIGH,
                title="Let's Break This Down",
                message="This is challenging. Would you like to see a worked example or try an easier version first?",
                action="Show Easier Version",
                action_data={"reduce_difficulty": True},
                dismissible=True,
                follow_ups=["Show worked example", "Review prerequisites"],
            )

        # SEVERE
        return Intervention(
            intervention_id=str(uuid.uuid4())[:8],
            intervention_type=InterventionType.BREAK_SUGGESTION,
            priority=InterventionPriority.CRITICAL,
            title="Time for a Break?",
            message=self._get_random_message(InterventionType.BREAK_SUGGESTION, context),
            action="Take a Break",
            action_data={"start_break": True, "duration_minutes": 5},
            follow_ups=["Try a different topic", "Review what you've learned"],
        )

    def _create_cognitive_load_intervention(
        self,
        cognitive_load_score: float,
        cognitive_load_level: str,
        context: Dict
    ) -> Optional[Intervention]:
        """Create intervention for cognitive overload"""
        import uuid

        if cognitive_load_level == "overload":
            return Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                intervention_type=InterventionType.SIMPLIFY_CONTENT,
                priority=InterventionPriority.HIGH,
                title="Let's Simplify",
                message="This might be too complex right now. Would you like a simpler explanation?",
                action="Simplify",
                action_data={"simplify": True},
            )
        elif cognitive_load_level == "low":
            # Under-challenged
            return Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                intervention_type=InterventionType.INCREASE_DIFFICULTY,
                priority=InterventionPriority.LOW,
                title="Ready for More?",
                message="You're doing great! Ready to try something more challenging?",
                action="Challenge Me",
                action_data={"increase_difficulty": True},
                dismissible=True,
            )

        return None

    def _create_metacognition_intervention(
        self,
        calibration_level: CalibrationLevel,
        context: Dict
    ) -> Optional[Intervention]:
        """Create metacognitive intervention"""
        import uuid

        if calibration_level == CalibrationLevel.OVERCONFIDENT:
            return Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                intervention_type=InterventionType.CALIBRATION_FEEDBACK,
                priority=InterventionPriority.MEDIUM,
                title="Check Your Understanding",
                message="Before moving on, try explaining this concept without looking at the material.",
                action="Self-Test",
                action_data={"trigger_self_test": True},
            )
        elif calibration_level == CalibrationLevel.UNDERCONFIDENT:
            return Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                intervention_type=InterventionType.ENCOURAGEMENT,
                priority=InterventionPriority.LOW,
                title="You Know More Than You Think!",
                message="Your performance shows you understand this better than you feel. Trust yourself!",
                dismissible=True,
            )

        return None

    def _create_progress_intervention(
        self,
        learner_state: LearnerState,
        context: Dict
    ) -> Optional[Intervention]:
        """Create progress-based intervention"""
        import uuid

        # Celebrate milestones
        if learner_state.concepts_mastered_today > 0 and learner_state.concepts_mastered_today % 3 == 0:
            return Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                intervention_type=InterventionType.PROGRESS_REMINDER,
                priority=InterventionPriority.LOW,
                title="Milestone Reached! 🎉",
                message=f"You've mastered {learner_state.concepts_mastered_today} concepts today! Great work!",
                dismissible=True,
                display_duration_seconds=8,
            )

        # Long session reminder
        if learner_state.session_duration_minutes > 45:
            return Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                intervention_type=InterventionType.BREAK_SUGGESTION,
                priority=InterventionPriority.MEDIUM,
                title="You've Been Working Hard!",
                message="You've been studying for 45+ minutes. A short break helps retention!",
                action="Take Break",
                action_data={"suggest_break": True},
            )

        return None

    def decide_intervention(
        self,
        learner_state: LearnerState,
        events: List[InteractionEvent],
        context: Optional[Dict] = None
    ) -> InterventionDecision:
        """
        Decide whether and how to intervene based on learner state

        Args:
            learner_state: Current state of the learner
            events: Recent interaction events
            context: Additional context

        Returns:
            InterventionDecision with intervention if appropriate
        """
        context = context or {}
        context.update({
            "mastered_count": learner_state.concepts_mastered_today,
            "minutes": int(learner_state.time_on_task_minutes),
        })

        # Check global cooldown
        if learner_state.last_intervention_time:
            if datetime.utcnow() - learner_state.last_intervention_time < self.min_cooldown:
                return InterventionDecision(
                    should_intervene=False,
                    intervention=None,
                    reason="global_cooldown",
                    cooldown_seconds=int(
                        (self.min_cooldown - (datetime.utcnow() - learner_state.last_intervention_time)).total_seconds()
                    )
                )

        # Collect potential interventions
        candidates: List[Tuple[Intervention, str]] = []

        # 1. Check frustration (highest priority)
        frustration_intervention = self._create_frustration_intervention(
            learner_state.frustration, context
        )
        if frustration_intervention:
            candidates.append((frustration_intervention, "frustration"))

        # 2. Check cognitive load
        load_intervention = self._create_cognitive_load_intervention(
            learner_state.cognitive_load_score,
            learner_state.cognitive_load_level,
            context
        )
        if load_intervention:
            candidates.append((load_intervention, "cognitive_load"))

        # 3. Check metacognition (if not too recent)
        if not self._check_cooldown(
            learner_state.user_id,
            InterventionType.CALIBRATION_FEEDBACK,
            learner_state.last_intervention_time
        ):
            meta_intervention = self._create_metacognition_intervention(
                learner_state.calibration_level, context
            )
            if meta_intervention:
                candidates.append((meta_intervention, "metacognition"))

        # 4. Check progress milestones
        progress_intervention = self._create_progress_intervention(learner_state, context)
        if progress_intervention:
            candidates.append((progress_intervention, "progress"))

        # No interventions needed
        if not candidates:
            return InterventionDecision(
                should_intervene=False,
                intervention=None,
                reason="no_intervention_needed"
            )

        # Sort by priority
        priority_order = {
            InterventionPriority.CRITICAL: 0,
            InterventionPriority.HIGH: 1,
            InterventionPriority.MEDIUM: 2,
            InterventionPriority.LOW: 3,
        }
        candidates.sort(key=lambda x: priority_order.get(x[0].priority, 4))

        # Select top intervention
        selected, reason = candidates[0]

        # Don't intervene for LOW priority with probability
        if selected.priority == InterventionPriority.LOW:
            if random.random() > self.intervention_frequency:
                return InterventionDecision(
                    should_intervene=False,
                    intervention=None,
                    reason="low_priority_skipped"
                )

        # Record intervention
        if learner_state.user_id not in self.intervention_history:
            self.intervention_history[learner_state.user_id] = []
        self.intervention_history[learner_state.user_id].append(selected)

        return InterventionDecision(
            should_intervene=True,
            intervention=selected,
            reason=reason,
            alternative_interventions=[c[0] for c in candidates[1:3]],
            cooldown_seconds=self.COOLDOWNS.get(selected.intervention_type, 60)
        )

    def get_intervention_effectiveness(
        self,
        user_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness of interventions for a user

        Returns:
            Statistics on intervention effectiveness
        """
        history = self.intervention_history.get(user_id, [])

        if not history:
            return {"message": "No intervention history"}

        # Count by type
        type_counts = {}
        for intervention in history:
            itype = intervention.intervention_type.value
            type_counts[itype] = type_counts.get(itype, 0) + 1

        return {
            "total_interventions": len(history),
            "by_type": type_counts,
            "most_common": max(type_counts, key=type_counts.get) if type_counts else None,
        }


# Global instance
intervention_engine = InterventionEngine()


def get_intervention_engine() -> InterventionEngine:
    """Dependency injection"""
    return intervention_engine
