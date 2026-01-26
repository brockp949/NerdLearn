"""
Structural Causal Model (SCM) for Educational Learning

Implements Pearl's SCM framework for formalizing the learning process as a causal graph.
This enables counterfactual reasoning via the Abduction-Action-Prediction process.

Key Components:
- ExogenousVariables (U): Unobserved background factors (aptitude, mood, noise)
- EndogenousVariables (V): Observable system variables (mastery, performance, actions)
- StructuralEquations (F): Deterministic mechanisms defining variable relationships

The SCM wraps the TD-BKT model to provide:
1. Forward simulation (prediction)
2. Intervention via do-calculus: do(X=x)
3. Counterfactual computation for "what-if" questions

Mathematical Foundation:
    M = <U, V, F>

    V = {L_t, Obs_t, A_t, Δt}
        L_t: Latent mastery at time t
        Obs_t: Observed performance (correct/incorrect)
        A_t: Learning action taken
        Δt: Time elapsed since last interaction

    U = {U_student, U_content, U_noise}
        U_student: Innate aptitude, current mood
        U_content: Question difficulty variability
        U_noise: Random measurement error

    F:
        L_t = f_L(L_{t-1}, A_{t-1}, Δt, U_student)
        Obs_t = f_Obs(L_t, U_content, U_noise)

References:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Counterfactual Explanations for Learning Paths (NerdLearn Research)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import math
import numpy as np
import logging

from app.adaptive.td_bkt.temporal_difference_bkt import (
    TemporalDifferenceBKT,
    TDBKTConfig,
    BeliefState,
    ConceptState,
)

logger = logging.getLogger(__name__)


class VariableType(str, Enum):
    """Types of variables in the SCM"""
    EXOGENOUS = "exogenous"      # Unobserved background factors
    ENDOGENOUS = "endogenous"    # Observable system variables


class ActionType(str, Enum):
    """Types of learning actions"""
    PRACTICE = "practice"        # Direct practice/quiz
    REVIEW = "review"            # Review/study material
    HINT = "hint"                # Used a hint
    VIDEO = "video"              # Watched instructional video
    SKIP = "skip"                # Skipped the concept
    PREREQUISITE = "prerequisite"  # Reviewed prerequisite


@dataclass
class ExogenousVariables:
    """
    Exogenous variables U representing unobserved background factors.

    These are the "noise" terms that introduce stochasticity into the system.
    Crucially, in counterfactual reasoning, U is inferred from observations
    and held fixed when simulating alternative scenarios.

    Attributes:
        u_student: Innate aptitude + current state factor [-1, 1]
            Positive = higher than average learning ability
            Negative = lower than average (fatigue, distraction)
        u_content: Content difficulty variability [-1, 1]
            Positive = question was easier than calibrated
            Negative = question was harder than calibrated
        u_noise: Random measurement/sensor noise [-1, 1]
            Captures unexplained variance
    """
    u_student: float = 0.0
    u_content: float = 0.0
    u_noise: float = 0.0

    def __post_init__(self):
        # Clamp to valid range
        self.u_student = max(-1.0, min(1.0, self.u_student))
        self.u_content = max(-1.0, min(1.0, self.u_content))
        self.u_noise = max(-1.0, min(1.0, self.u_noise))

    @classmethod
    def sample_prior(cls) -> "ExogenousVariables":
        """Sample from prior distribution (standard normal truncated to [-1, 1])"""
        return cls(
            u_student=np.clip(np.random.normal(0, 0.3), -1, 1),
            u_content=np.clip(np.random.normal(0, 0.2), -1, 1),
            u_noise=np.clip(np.random.normal(0, 0.1), -1, 1),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "u_student": self.u_student,
            "u_content": self.u_content,
            "u_noise": self.u_noise,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ExogenousVariables":
        return cls(**d)


@dataclass
class EndogenousVariables:
    """
    Endogenous variables V with explicit causal parents.

    These are the variables within the system's model that have
    causal relationships defined by the structural equations.

    Attributes:
        L_t: Latent mastery probability at time t [0, 1]
        Obs_t: Observed performance outcome
        A_t: Action/intervention taken (concept_id, action_type)
        Delta_t: Time elapsed since last interaction (hours)
        response_time_ms: Optional response time sensor data
        concept_id: Which concept this observation is for
        timestamp: When the observation occurred
    """
    L_t: float                          # Mastery probability
    Obs_t: Optional[bool] = None        # Observed outcome (correct/incorrect)
    A_t: Optional[str] = None           # Action taken
    Delta_t: float = 0.0                # Time elapsed (hours)
    response_time_ms: Optional[int] = None
    concept_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    # Additional context
    action_type: ActionType = ActionType.PRACTICE
    half_life_hours: float = 24.0       # Memory strength

    def __post_init__(self):
        self.L_t = max(0.0, min(1.0, self.L_t))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "L_t": self.L_t,
            "Obs_t": self.Obs_t,
            "A_t": self.A_t,
            "Delta_t": self.Delta_t,
            "response_time_ms": self.response_time_ms,
            "concept_id": self.concept_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "action_type": self.action_type.value,
            "half_life_hours": self.half_life_hours,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EndogenousVariables":
        d = d.copy()
        if d.get("timestamp"):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        if d.get("action_type"):
            d["action_type"] = ActionType(d["action_type"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CausalEdge:
    """Represents a directed edge in the causal graph"""
    source: str
    target: str
    edge_type: str = "directed"  # directed, bi-directed (confounded)
    strength: float = 1.0        # Edge weight/strength

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "strength": self.strength,
        }


class StructuralEquations:
    """
    Structural equations F defining the causal mechanisms.

    These functions describe deterministically how variables interact.
    Each endogenous variable has a structural equation:
        V_i = f_i(Parents(V_i), U_i)

    The key insight is that the error term U is not just noise to average out;
    it represents the specific, unobserved reality for that student.
    """

    def __init__(self, td_bkt_config: Optional[TDBKTConfig] = None):
        """
        Initialize structural equations with BKT parameters.

        Args:
            td_bkt_config: Configuration for TD-BKT parameters
        """
        self.config = td_bkt_config or TDBKTConfig()

    def mastery_transition(
        self,
        L_prev: float,
        action_type: ActionType,
        delta_t_hours: float,
        u_student: float,
        half_life_hours: float = 24.0,
    ) -> Tuple[float, float]:
        """
        Structural equation for mastery transition.

        Implements:
        1. Half-life decay: L_{decay} = L_{prev} * 2^(-Δt/h)
        2. Learning gain: L_new = L_{decay} + P(T) * (1 - L_{decay}) * (1 + u_student)
        3. Spacing effect: Bonus for optimally spaced practice

        Args:
            L_prev: Previous mastery probability
            action_type: Type of learning action
            delta_t_hours: Time elapsed since last interaction
            u_student: Student-specific exogenous factor
            half_life_hours: Memory half-life

        Returns:
            (new_mastery, new_half_life_hours)
        """
        # Step 1: Apply forgetting (half-life decay)
        if delta_t_hours > 0 and half_life_hours > 0:
            decay_factor = math.pow(2, -delta_t_hours / half_life_hours)
        else:
            decay_factor = 1.0

        L_decayed = L_prev * decay_factor

        # Step 2: Apply learning based on action type
        base_learning_rate = self.config.p_t  # Base transition probability

        # Modulate by action type
        action_multipliers = {
            ActionType.PRACTICE: 1.0,
            ActionType.REVIEW: 0.7,
            ActionType.HINT: 0.3,
            ActionType.VIDEO: 0.5,
            ActionType.SKIP: 0.0,
            ActionType.PREREQUISITE: 0.4,
        }
        action_mult = action_multipliers.get(action_type, 0.5)

        # Modulate by student factor (aptitude/state)
        # u_student in [-1, 1] maps to [0.5, 1.5] multiplier
        student_mult = 1.0 + 0.5 * u_student

        # Spacing effect: bonus for optimally spaced practice (around 1 day)
        optimal_spacing_hours = 24.0
        if delta_t_hours > 0:
            spacing_factor = math.exp(-abs(math.log(delta_t_hours / optimal_spacing_hours)) / 2)
            spacing_bonus = 0.1 * spacing_factor
        else:
            spacing_bonus = 0.0

        # Effective learning rate
        effective_lr = base_learning_rate * action_mult * student_mult + spacing_bonus
        effective_lr = max(0.0, min(0.5, effective_lr))  # Cap at 50% per opportunity

        # Apply learning transition
        if action_type != ActionType.SKIP:
            L_new = L_decayed + effective_lr * (1 - L_decayed)
        else:
            L_new = L_decayed

        # Update half-life based on successful practice
        # Practice strengthens memory
        if action_type in [ActionType.PRACTICE, ActionType.REVIEW]:
            # Spacing bonus for memory strength
            spacing_hl_bonus = 1.0 + min(delta_t_hours / 48.0, 1.0) * 0.3
            new_half_life = min(
                half_life_hours * self.config.half_life_growth_factor * spacing_hl_bonus,
                self.config.max_half_life_days * 24
            )
        else:
            new_half_life = half_life_hours

        return (max(0.0, min(1.0, L_new)), new_half_life)

    def observation_likelihood(
        self,
        L_t: float,
        u_content: float,
        u_noise: float,
    ) -> Tuple[float, float]:
        """
        Compute observation likelihood P(Correct | L_t, U).

        Standard BKT observation model with exogenous modulation:
            P(Correct) = L_t * (1 - P_slip) + (1 - L_t) * P_guess

        Where slip/guess are modulated by content difficulty and noise.

        Args:
            L_t: Current mastery probability
            u_content: Content difficulty variability
            u_noise: Random noise factor

        Returns:
            (p_correct, p_slip_adjusted)
        """
        # Base slip and guess from config
        p_s_base = self.config.p_s
        p_g_base = self.config.p_g

        # Adjust slip by content difficulty and noise
        # Harder content (negative u_content) increases slip
        p_s_adjusted = p_s_base * (1 - 0.5 * u_content)
        p_s_adjusted += 0.05 * abs(u_noise)  # Noise adds uncertainty
        p_s_adjusted = max(0.01, min(0.5, p_s_adjusted))

        # Adjust guess by content difficulty
        # Harder content reduces guessing success
        p_g_adjusted = p_g_base * (1 + 0.3 * u_content)
        p_g_adjusted = max(0.01, min(0.5, p_g_adjusted))

        # BKT observation model
        p_correct = L_t * (1 - p_s_adjusted) + (1 - L_t) * p_g_adjusted

        return (p_correct, p_s_adjusted)

    def correction_update(
        self,
        L_prior: float,
        observed_correct: bool,
        p_slip: float,
        p_guess: float,
    ) -> float:
        """
        Bayesian correction step after observation.

        P(L | Obs) using Bayes' rule:
            P(L | Correct) = P(Correct | L) * P(L) / P(Correct)

        Args:
            L_prior: Prior mastery before observation
            observed_correct: Whether response was correct
            p_slip: Slip probability
            p_guess: Guess probability

        Returns:
            Posterior mastery probability
        """
        if observed_correct:
            p_obs = L_prior * (1 - p_slip) + (1 - L_prior) * p_guess
            if p_obs > 0:
                L_posterior = L_prior * (1 - p_slip) / p_obs
            else:
                L_posterior = L_prior
        else:
            p_obs = L_prior * p_slip + (1 - L_prior) * (1 - p_guess)
            if p_obs > 0:
                L_posterior = L_prior * p_slip / p_obs
            else:
                L_posterior = L_prior

        # Apply learning transition
        L_final = L_posterior + (1 - L_posterior) * self.config.p_t

        return max(0.0, min(1.0, L_final))


class StructuralCausalModel:
    """
    Educational Structural Causal Model: M = <U, V, F>

    Wraps TD-BKT with formal causal semantics to enable:
    1. Forward simulation (standard prediction)
    2. Intervention via do-calculus: do(X=x)
    3. Counterfactual computation via abduction-action-prediction

    The key distinction from standard prediction:
    - Prediction: P(Y | X=x) - observational distribution
    - Intervention: P(Y | do(X=x)) - causal effect
    - Counterfactual: P(Y_x | X=x', Y=y') - what would Y be if we do X=x, given we observed Y=y'

    Usage:
        scm = StructuralCausalModel(td_bkt)

        # Forward simulation
        trajectory = scm.forward_simulate(belief_state, actions)

        # Intervention
        new_state = scm.intervene("Delta_t", 10.0, current_state)

        # Counterfactual (via CounterfactualEngine)
        # 1. Abduction: infer U from observation
        # 2. Action: intervene on SCM
        # 3. Prediction: simulate with inferred U
    """

    def __init__(
        self,
        td_bkt: Optional[TemporalDifferenceBKT] = None,
        causal_graph: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize SCM.

        Args:
            td_bkt: TD-BKT instance for state estimation
            causal_graph: Custom causal graph (adjacency list)
        """
        self.td_bkt = td_bkt or TemporalDifferenceBKT()
        self.equations = StructuralEquations(self.td_bkt.config)
        self.causal_graph = causal_graph or self._default_learning_graph()

        logger.info("StructuralCausalModel initialized")

    def _default_learning_graph(self) -> Dict[str, List[str]]:
        """
        Default causal graph for learning.

        Structure:
            L_{t-1} ──┬──> L_t ──> Obs_t
                      │     ^        ^
            A_{t-1} ──┘     │        │
                            │        │
            Δt ─────────────┘        │
                                     │
            U_student ───────────────┤
            U_content ───────────────┤
            U_noise ─────────────────┘

        Returns:
            Adjacency list representation of causal graph
        """
        return {
            # Endogenous variable parents
            "L_t": ["L_t_prev", "A_t_prev", "Delta_t", "U_student"],
            "Obs_t": ["L_t", "U_content", "U_noise"],
            "A_t": [],  # Action is chosen by policy (external)
            "Delta_t": [],  # Time is observed (external)

            # Exogenous have no parents by definition
            "U_student": [],
            "U_content": [],
            "U_noise": [],

            # Previous state variables
            "L_t_prev": [],
            "A_t_prev": [],
        }

    def get_variable_parents(self, variable: str) -> List[str]:
        """Get causal parents of a variable"""
        return self.causal_graph.get(variable, [])

    def intervene(
        self,
        variable: str,
        value: Any,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply do-operator intervention: do(variable = value)

        This removes all incoming edges to the variable and fixes its value.
        The intervention breaks the causal link between the variable and its parents.

        Args:
            variable: Variable to intervene on
            value: Fixed value for the intervention
            current_state: Current SCM state

        Returns:
            Modified state after intervention
        """
        # Create copy of state
        new_state = current_state.copy()

        # Set the intervened variable to fixed value
        new_state[variable] = value
        new_state[f"{variable}_intervened"] = True

        logger.debug(f"Intervention: do({variable} = {value})")

        return new_state

    def compute_endogenous(
        self,
        state: Dict[str, Any],
        exogenous: ExogenousVariables,
    ) -> EndogenousVariables:
        """
        Compute endogenous variables from state and exogenous.

        This applies the structural equations to compute the values
        of endogenous variables given their parents and exogenous noise.

        Args:
            state: Current SCM state with parent variable values
            exogenous: Exogenous variable values

        Returns:
            Computed endogenous variables
        """
        # Extract parent values
        L_prev = state.get("L_t_prev", 0.1)
        action_type = ActionType(state.get("action_type", "practice"))
        delta_t = state.get("Delta_t", 0.0)
        half_life = state.get("half_life_hours", 24.0)

        # Check if L_t was intervened (fixed value)
        if state.get("L_t_intervened"):
            L_new = state["L_t"]
            new_half_life = half_life
        else:
            # Apply mastery transition equation
            L_new, new_half_life = self.equations.mastery_transition(
                L_prev=L_prev,
                action_type=action_type,
                delta_t_hours=delta_t,
                u_student=exogenous.u_student,
                half_life_hours=half_life,
            )

        # Compute observation probability
        p_correct, p_slip = self.equations.observation_likelihood(
            L_t=L_new,
            u_content=exogenous.u_content,
            u_noise=exogenous.u_noise,
        )

        return EndogenousVariables(
            L_t=L_new,
            Obs_t=None,  # Not yet observed
            A_t=state.get("A_t"),
            Delta_t=delta_t,
            concept_id=state.get("concept_id"),
            action_type=action_type,
            half_life_hours=new_half_life,
        )

    def simulate_observation(
        self,
        endogenous: EndogenousVariables,
        exogenous: ExogenousVariables,
    ) -> bool:
        """
        Simulate observation outcome (correct/incorrect).

        Samples from P(Obs_t | L_t, U).

        Args:
            endogenous: Current endogenous variables
            exogenous: Exogenous variables

        Returns:
            Simulated observation (True = correct)
        """
        p_correct, _ = self.equations.observation_likelihood(
            L_t=endogenous.L_t,
            u_content=exogenous.u_content,
            u_noise=exogenous.u_noise,
        )

        return np.random.random() < p_correct

    def forward_simulate(
        self,
        initial_belief: BeliefState,
        action_sequence: List[Tuple[str, datetime, ActionType]],
        exogenous: Optional[ExogenousVariables] = None,
        simulate_observations: bool = True,
    ) -> List[EndogenousVariables]:
        """
        Simulate forward trajectory given an action sequence.

        This is standard prediction: P(Y | X=x).

        Args:
            initial_belief: Starting belief state
            action_sequence: List of (concept_id, timestamp, action_type)
            exogenous: Fixed exogenous variables (samples from prior if None)
            simulate_observations: Whether to sample observations

        Returns:
            List of endogenous variable states for each timestep
        """
        if exogenous is None:
            exogenous = ExogenousVariables.sample_prior()

        trajectory = []
        current_masteries = {
            cid: state.mastery_probability
            for cid, state in initial_belief.concept_states.items()
        }
        current_half_lives = {
            cid: state.half_life_days * 24  # Convert to hours
            for cid, state in initial_belief.concept_states.items()
        }
        last_timestamps = {
            cid: state.last_interaction_at
            for cid, state in initial_belief.concept_states.items()
        }

        for concept_id, timestamp, action_type in action_sequence:
            # Compute time delta
            if concept_id in last_timestamps and last_timestamps[concept_id]:
                delta_t = (timestamp - last_timestamps[concept_id]).total_seconds() / 3600
            else:
                delta_t = 0.0

            # Build SCM state
            state = {
                "L_t_prev": current_masteries.get(concept_id, 0.1),
                "A_t": concept_id,
                "Delta_t": delta_t,
                "action_type": action_type.value,
                "half_life_hours": current_half_lives.get(concept_id, 24.0),
                "concept_id": concept_id,
            }

            # Compute new endogenous variables
            endo = self.compute_endogenous(state, exogenous)
            endo.timestamp = timestamp

            # Optionally simulate observation
            if simulate_observations:
                endo.Obs_t = self.simulate_observation(endo, exogenous)

            trajectory.append(endo)

            # Update state for next iteration
            current_masteries[concept_id] = endo.L_t
            current_half_lives[concept_id] = endo.half_life_hours
            last_timestamps[concept_id] = timestamp

        return trajectory

    def state_from_belief(
        self,
        belief_state: BeliefState,
        concept_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Extract SCM state from belief state for a specific concept.

        Args:
            belief_state: Current belief state
            concept_id: Concept to extract state for
            timestamp: Current timestamp

        Returns:
            SCM state dictionary
        """
        timestamp = timestamp or datetime.now()

        if concept_id in belief_state.concept_states:
            concept_state = belief_state.concept_states[concept_id]
            L_prev = concept_state.mastery_probability
            half_life = concept_state.half_life_days * 24

            if concept_state.last_interaction_at:
                delta_t = (timestamp - concept_state.last_interaction_at).total_seconds() / 3600
            else:
                delta_t = 0.0
        else:
            L_prev = 0.1
            half_life = 24.0
            delta_t = 0.0

        return {
            "L_t_prev": L_prev,
            "A_t": concept_id,
            "Delta_t": delta_t,
            "half_life_hours": half_life,
            "concept_id": concept_id,
            "action_type": ActionType.PRACTICE.value,
        }

    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the causal graph structure"""
        return {
            "variables": list(self.causal_graph.keys()),
            "edges": [
                {"source": parent, "target": child}
                for child, parents in self.causal_graph.items()
                for parent in parents
            ],
            "num_endogenous": 4,  # L_t, Obs_t, A_t, Delta_t
            "num_exogenous": 3,   # U_student, U_content, U_noise
        }
