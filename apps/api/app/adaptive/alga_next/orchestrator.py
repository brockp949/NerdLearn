"""
ALGA-Next Orchestrator: Adaptive Learning via Generative Allocation

Main orchestration service that integrates all ALGA-Next components:
1. MouStress Analyzer - Mouse dynamics → cognitive state
2. MMSAF-Net - Multi-modal telemetry fusion → User State Vector
3. Hybrid LinUCB - Contextual bandit → modality selection
4. Attention Transfer - Cold-start handling → cross-modality prediction
5. Composite Reward - Pedagogical reward calculation
6. Generative UI - SDUI schema generation

The orchestrator implements the closed feedback loop:
    Sensors → Feature Engineering → Bandit Core → Generative UI → Feedback

This is the main entry point for adaptive modality selection in NerdLearn.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
import logging

from .hybrid_linucb import (
    HybridLinUCB,
    ModalityArm,
    ContextVector,
    ModalityPolicy,
    Modality,
    create_modality_arms,
)
from .mmsaf_net import (
    MMSAFNet,
    UserStateVector,
    BehavioralFeatures,
    ContextualFeatures,
    ContentFeatures,
)
from .attention_transfer import (
    AttentionTransferNetwork,
    UserObservation,
    ModalityType,
    TransferPredictions,
)
from .reward_function import (
    CompositeRewardFunction,
    RewardComponents,
    RewardConfig,
    RewardObjective,
)
from .generative_ui import (
    GenerativeUIRegistry,
    SDUISchema,
    ScaffoldingLevel,
    AtomicContentUnit,
)
from .mouse_stress import (
    MouStressAnalyzer,
    MouStressResult,
    LearnerState,
)

logger = logging.getLogger(__name__)


@dataclass
class MouseEvent:
    """Raw mouse event from telemetry"""
    x: int
    y: int
    timestamp_ms: float
    event_type: str = "move"


@dataclass
class TelemetrySnapshot:
    """Snapshot of telemetry data for processing"""
    session_id: str
    user_id: str
    timestamp: datetime

    # Mouse events (last N events)
    mouse_events: List[MouseEvent] = field(default_factory=list)

    # Dwell and engagement
    dwell_time_ms: float = 0.0
    scroll_depth: float = 0.0
    click_count: int = 0
    interaction_count: int = 0

    # Session context
    session_duration_minutes: float = 0.0
    cards_completed: int = 0
    recent_success_rate: float = 0.5

    # Current content
    current_concept_id: str = ""
    current_content_id: str = ""
    current_modality: str = "text"


@dataclass
class ModalitySelectionResult:
    """Result of the modality selection process"""
    # Selected modality and content
    selected_modality: Modality
    selected_content_id: str
    confidence: float

    # User state
    user_state: UserStateVector
    learner_state: LearnerState

    # SDUI schema for rendering
    ui_schema: SDUISchema

    # Explanation and metadata
    explanation: str
    alternatives: List[Dict[str, Any]]
    exploration_bonus: float

    # Intervention if needed
    intervention: Optional[Dict[str, Any]] = None


class ALGANextOrchestrator:
    """
    Main orchestrator for ALGA-Next adaptive modality selection

    Coordinates all components to:
    1. Process telemetry into cognitive state
    2. Select optimal modality using contextual bandit
    3. Handle cold-start with attention transfer
    4. Generate adaptive UI schema
    5. Calculate rewards for learning loop
    """

    def __init__(
        self,
        reward_objective: RewardObjective = RewardObjective.BALANCED,
        exploration_rate: float = 1.0,
    ):
        # Initialize components
        self.mmsaf_net = MMSAFNet(use_torch=False)  # Use numpy for portability

        self.linucb = HybridLinUCB(
            shared_dim=12,
            arm_dim=5,
            interaction_dim=9,
            alpha=exploration_rate,
        )

        self.attention_transfer = AttentionTransferNetwork(
            modalities=[ModalityType(m.value) for m in Modality],
        )

        self.reward_function = CompositeRewardFunction(
            config=RewardConfig.for_objective(reward_objective)
        )

        self.ui_registry = GenerativeUIRegistry()

        # Per-user MouStress analyzers
        self.mouse_analyzers: Dict[str, MouStressAnalyzer] = {}

        # State caches
        self.user_states: Dict[str, UserStateVector] = {}
        self.session_contexts: Dict[str, TelemetrySnapshot] = {}

        logger.info(f"ALGANextOrchestrator initialized with {reward_objective.value} objective")

    def get_mouse_analyzer(self, user_id: str) -> MouStressAnalyzer:
        """Get or create MouStress analyzer for user"""
        if user_id not in self.mouse_analyzers:
            self.mouse_analyzers[user_id] = MouStressAnalyzer()
        return self.mouse_analyzers[user_id]

    async def process_telemetry(
        self,
        telemetry: TelemetrySnapshot,
    ) -> UserStateVector:
        """
        Process telemetry snapshot into User State Vector

        Pipeline:
        1. Feed mouse events to MouStress analyzer
        2. Extract behavioral features
        3. Combine with contextual features
        4. Run MMSAF-Net fusion
        """
        user_id = telemetry.user_id

        # Get mouse analyzer
        analyzer = self.get_mouse_analyzer(user_id)

        # Add mouse events
        for event in telemetry.mouse_events:
            analyzer.add_event(
                x=event.x,
                y=event.y,
                timestamp_ms=event.timestamp_ms,
                event_type=event.event_type,
            )

        # Run MouStress analysis
        mouse_result = analyzer.analyze()

        # Create behavioral features from MouStress
        behavioral = BehavioralFeatures(
            mouse_velocity=mouse_result.trajectory_analysis.avg_velocity,
            mouse_jitter=mouse_result.kinematic_stiffness.tremor_amplitude,
            velocity_std=mouse_result.trajectory_analysis.velocity_std,
            click_rate=telemetry.click_count / max(1, telemetry.session_duration_minutes),
            idle_time_ratio=len(mouse_result.trajectory_analysis.idle_periods) / max(1, len(telemetry.mouse_events)),
            micro_hesitation_rate=mouse_result.trajectory_analysis.micro_hesitation_count,
            dwell_time_factor=min(1.0, telemetry.dwell_time_ms / 60000),  # Normalize to 1 min
            straightness_ratio=mouse_result.trajectory_analysis.straightness_ratio,
            curvature_entropy=mouse_result.trajectory_analysis.curvature_entropy,
            scroll_velocity=0.0,  # Would come from scroll telemetry
            scroll_depth=telemetry.scroll_depth,
        )

        # Create contextual features
        contextual = ContextualFeatures(
            time_of_day=datetime.now().hour / 24,
            day_of_week=datetime.now().weekday(),
            session_duration_minutes=telemetry.session_duration_minutes,
            device_type="desktop",  # Would come from client
            session_card_count=telemetry.cards_completed,
            session_success_rate=telemetry.recent_success_rate,
        )

        # Create content features (from current content)
        content = ContentFeatures(
            complexity_score=0.5,  # Would come from content metadata
            modality_type=telemetry.current_modality,
            content_length_minutes=5.0,
        )

        # Run MMSAF-Net fusion
        user_state = self.mmsaf_net.infer_state(behavioral, contextual, content)

        # Cache state
        self.user_states[user_id] = user_state
        self.session_contexts[telemetry.session_id] = telemetry

        return user_state

    async def select_modality(
        self,
        user_id: str,
        concept_id: str,
        available_content: List[Dict[str, Any]],
        telemetry: Optional[TelemetrySnapshot] = None,
        device: str = "desktop",
    ) -> ModalitySelectionResult:
        """
        Select optimal modality and generate UI schema

        Main decision flow:
        1. Process telemetry → User State Vector
        2. Check cold-start with Attention Transfer
        3. Run Hybrid LinUCB selection
        4. Determine scaffolding level
        5. Generate SDUI schema
        """
        # Process telemetry if provided
        if telemetry:
            user_state = await self.process_telemetry(telemetry)
        else:
            user_state = self.user_states.get(user_id, UserStateVector())

        # Get learner state from mouse analyzer
        analyzer = self.get_mouse_analyzer(user_id)
        mouse_result = analyzer.analyze()
        learner_state = mouse_result.learner_state

        # Create context vector for LinUCB
        context = ContextVector(
            fatigue_level=user_state.fatigue_level,
            attention_level=user_state.focus_level,
            cognitive_capacity=user_state.cognitive_capacity,
            device_type=device,
            time_of_day=datetime.now().hour / 24,
            current_difficulty=0.5,  # Would come from concept metadata
            topic_familiarity=0.5,  # Would come from user profile
            session_duration_minutes=telemetry.session_duration_minutes if telemetry else 0,
            recent_success_rate=telemetry.recent_success_rate if telemetry else 0.5,
        )

        # Create modality arms from available content
        arms = create_modality_arms(available_content)

        # Check cold-start: do we have data for this user?
        transfer_preds = self.attention_transfer.predict(user_id)

        # If cold-start (low confidence), boost exploration
        min_confidence = min(
            p.confidence for p in transfer_preds.predictions.values()
        ) if transfer_preds.predictions else 0.0

        if min_confidence < 0.3:
            # Cold-start: increase exploration
            self.linucb.set_exploration_rate(2.0)
        else:
            self.linucb.set_exploration_rate(1.0)

        # Run LinUCB selection
        policy = self.linucb.select_arm(context, arms)

        # Determine scaffolding level based on learner state
        scaffolding = self._determine_scaffolding(
            learner_state, user_state, context
        )

        # Check for intervention needs
        intervention = self._check_intervention(
            user_state, learner_state, telemetry
        )

        # Generate SDUI schema
        ui_schema = self.ui_registry.generate_schema(
            concept_id=concept_id,
            modality=policy.selected_arm.modality.value,
            scaffolding_level=scaffolding,
            fatigue_level=user_state.fatigue_level,
            device=device,
        )

        # Create result
        return ModalitySelectionResult(
            selected_modality=policy.selected_arm.modality,
            selected_content_id=policy.selected_arm.content_id,
            confidence=policy.confidence,
            user_state=user_state,
            learner_state=learner_state,
            ui_schema=ui_schema,
            explanation=policy.explanation,
            alternatives=[
                {"arm_id": arm_id, "score": score}
                for arm_id, score in policy.alternatives
            ],
            exploration_bonus=policy.exploration_bonus,
            intervention=intervention,
        )

    async def record_outcome(
        self,
        user_id: str,
        session_id: str,
        content_id: str,
        modality: str,
        outcome: Dict[str, Any],
    ):
        """
        Record learning outcome and update models

        Updates:
        1. Hybrid LinUCB with reward
        2. Attention Transfer with observation
        3. Reward function statistics
        """
        # Get context from cache
        telemetry = self.session_contexts.get(session_id)
        user_state = self.user_states.get(user_id, UserStateVector())

        # Create reward components
        components = RewardComponents(
            engagement_score=outcome.get("engagement_score", 0.5),
            dwell_time_ms=outcome.get("dwell_time_ms", 0),
            expected_dwell_ms=outcome.get("expected_dwell_ms", 60000),
            dwell_ratio=outcome.get("dwell_ratio", 1.0),
            scroll_completion=outcome.get("scroll_completion", 0.5),
            interaction_rate=outcome.get("interaction_rate", 0),
            assessment_score=outcome.get("assessment_score"),
            fatigue_level=user_state.fatigue_level,
            session_duration_minutes=telemetry.session_duration_minutes if telemetry else 0,
            modality_type=modality,
        )

        # Calculate composite reward
        reward, breakdown = self.reward_function.calculate(components)

        # Update LinUCB
        if telemetry:
            context = ContextVector(
                fatigue_level=user_state.fatigue_level,
                attention_level=user_state.focus_level,
                cognitive_capacity=user_state.cognitive_capacity,
                recent_success_rate=outcome.get("success", 0.5),
            )

            arm = ModalityArm(
                modality=Modality(modality),
                content_id=content_id,
            )

            self.linucb.update(context, arm, reward)

        # Update Attention Transfer
        observation = UserObservation(
            user_id=user_id,
            modality=ModalityType(modality),
            content_id=content_id,
            engagement_score=outcome.get("engagement_score", 0.5),
            completion_rate=outcome.get("completion_rate", 0.5),
            assessment_score=outcome.get("assessment_score"),
            fatigue_level=user_state.fatigue_level,
        )
        self.attention_transfer.observe(observation)

        logger.info(
            f"Recorded outcome for user {user_id}: "
            f"reward={reward:.3f}, modality={modality}"
        )

        return {
            "reward": reward,
            "breakdown": breakdown,
        }

    def _determine_scaffolding(
        self,
        learner_state: LearnerState,
        user_state: UserStateVector,
        context: ContextVector,
    ) -> ScaffoldingLevel:
        """Determine appropriate scaffolding level"""
        # High scaffolding for confusion/frustration
        if learner_state == LearnerState.CONFUSION:
            return ScaffoldingLevel.MODERATE
        elif learner_state == LearnerState.FRUSTRATION:
            return ScaffoldingLevel.INTENSIVE

        # Check confusion indicator
        if user_state.confusion_indicator > 0.6:
            return ScaffoldingLevel.MODERATE

        # Check recent success rate
        if context.recent_success_rate < 0.4:
            return ScaffoldingLevel.MODERATE
        elif context.recent_success_rate < 0.6:
            return ScaffoldingLevel.MINIMAL

        return ScaffoldingLevel.NONE

    def _check_intervention(
        self,
        user_state: UserStateVector,
        learner_state: LearnerState,
        telemetry: Optional[TelemetrySnapshot],
    ) -> Optional[Dict[str, Any]]:
        """Check if intervention is needed"""
        # Fatigue intervention
        if user_state.fatigue_level > 0.8:
            return {
                "type": "break_prompt",
                "message": "You've been working hard! Consider a 5-minute break.",
                "priority": "high",
                "action": "take_break",
            }

        # Frustration intervention
        if learner_state == LearnerState.FRUSTRATION:
            return {
                "type": "support_offer",
                "message": "This seems challenging. Would you like to try a different approach?",
                "priority": "high",
                "action": "change_modality",
            }

        # Distraction intervention
        if learner_state == LearnerState.DISTRACTION:
            return {
                "type": "attention_prompt",
                "message": "Still there? Ready to continue?",
                "priority": "low",
                "action": "continue",
            }

        # Long session intervention
        if telemetry and telemetry.session_duration_minutes > 45:
            return {
                "type": "break_prompt",
                "message": "You've been studying for 45+ minutes. Great focus! Time for a stretch?",
                "priority": "medium",
                "action": "take_break",
            }

        return None

    def register_content(self, unit: AtomicContentUnit):
        """Register content unit in the UI registry"""
        self.ui_registry.register_unit(unit)

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "linucb": self.linucb.get_statistics(),
            "reward_function": self.reward_function.get_statistics(),
            "transfer_matrix": self.attention_transfer.get_transfer_matrix(),
            "ui_registry": self.ui_registry.get_statistics(),
            "active_users": len(self.mouse_analyzers),
            "active_sessions": len(self.session_contexts),
        }


# Singleton instance
_orchestrator: Optional[ALGANextOrchestrator] = None


def get_orchestrator() -> ALGANextOrchestrator:
    """Get or create the ALGA-Next orchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ALGANextOrchestrator()
    return _orchestrator
