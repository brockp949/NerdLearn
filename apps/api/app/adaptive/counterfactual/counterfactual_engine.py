"""
Counterfactual Engine - Computing "What-If" Scenarios

Implements Pearl's 3-step counterfactual computation process:
1. Abduction: Infer exogenous U from observed outcome
2. Action: Apply do-calculus intervention on SCM
3. Prediction: Simulate alternative outcome with inferred U

This distinguishes counterfactuals from simple conditional predictions:
- Prediction P(Pass | Study+10): "What happens to average student who studies more?"
- Counterfactual P(Pass_{Study+10} | Obs=Fail): "What would happen to THIS student,
  who already failed, if they had studied 10 more minutes?"

The key insight is using the INFERRED exogenous variables (specific to this
student/context) rather than population averages.

Research Basis:
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
- Counterfactual Explanations for Learning Paths (NerdLearn Research)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import math
import numpy as np
from scipy import optimize
import logging

from app.adaptive.counterfactual.scm import (
    StructuralCausalModel,
    ExogenousVariables,
    EndogenousVariables,
    ActionType,
)
from app.adaptive.td_bkt.temporal_difference_bkt import (
    TemporalDifferenceBKT,
    BeliefState,
    ConceptState,
)
from app.adaptive.simulator.student_simulator import (
    StudentSimulator,
    StudentSimulatorConfig,
)

logger = logging.getLogger(__name__)


class CounterfactualType(str, Enum):
    """Types of counterfactual queries"""
    TEMPORAL = "temporal"          # "What if I had studied earlier/later?"
    EFFORT = "effort"              # "What if I had studied more/longer?"
    ACTION = "action"              # "What if I had chosen different activity?"
    SCHEDULING = "scheduling"      # "What if I had spaced practice differently?"
    PREREQUISITE = "prerequisite"  # "What if I had reviewed prerequisites?"


@dataclass
class CounterfactualQuery:
    """
    Query for counterfactual computation.

    Represents a "what-if" question about the learning process.

    Attributes:
        observed_outcome: The actual observed endogenous variables
        intervention: The counterfactual change to apply
            e.g., {"Delta_t": 10.0} means "add 10 hours of study time"
        query_type: Type of counterfactual query
        question: Natural language form of the question
        concept_id: Which concept the query is about
    """
    observed_outcome: EndogenousVariables
    intervention: Dict[str, Any]
    query_type: CounterfactualType = CounterfactualType.EFFORT
    question: Optional[str] = None
    concept_id: Optional[str] = None

    def __post_init__(self):
        if self.question is None:
            self.question = self._generate_question()

    def _generate_question(self) -> str:
        """Generate natural language question from intervention"""
        if "Delta_t" in self.intervention:
            delta = self.intervention["Delta_t"]
            if delta > 0:
                return f"What if you had studied {delta:.0f} hours more?"
            else:
                return f"What if you had studied {abs(delta):.0f} hours less?"
        elif "action_type" in self.intervention:
            return f"What if you had chosen {self.intervention['action_type']} instead?"
        elif "L_t_prev" in self.intervention:
            return "What if your prior mastery had been different?"
        else:
            return f"What if {list(self.intervention.keys())[0]} had been different?"


@dataclass
class CounterfactualResult:
    """
    Result of counterfactual computation.

    Contains both the original and counterfactual outcomes,
    the inferred exogenous variables, and explanation.

    Attributes:
        original_outcome: What actually happened
        counterfactual_outcome: What would have happened under intervention
        probability_change: Change in success/mastery probability
        inferred_exogenous: The inferred U values for this specific case
        explanation: Human-readable explanation
        confidence: Confidence in the counterfactual estimate [0, 1]
        alternative_interventions: Other interventions that could help
    """
    original_outcome: EndogenousVariables
    counterfactual_outcome: EndogenousVariables
    probability_change: float
    inferred_exogenous: ExogenousVariables
    explanation: str
    confidence: float
    alternative_interventions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original_outcome.to_dict(),
            "counterfactual": self.counterfactual_outcome.to_dict(),
            "probability_change": self.probability_change,
            "inferred_exogenous": self.inferred_exogenous.to_dict(),
            "explanation": self.explanation,
            "confidence": self.confidence,
            "alternative_interventions": self.alternative_interventions,
        }


@dataclass
class TrajectoryComparison:
    """Comparison of two learning trajectories"""
    original_trajectory: List[EndogenousVariables]
    counterfactual_trajectory: List[EndogenousVariables]
    final_mastery_original: float
    final_mastery_counterfactual: float
    improvement: float
    day30_retention_original: float
    day30_retention_counterfactual: float
    critical_divergence_point: Optional[int] = None  # Timestep where paths diverge


class CounterfactualEngine:
    """
    Engine for computing counterfactual explanations.

    Implements Pearl's 3-step counterfactual process:
    1. Abduction: P(U | Obs=observed) - infer background factors
    2. Action: do(X = counterfactual_value) - apply intervention
    3. Prediction: P(Y | do(X), U_inferred) - predict under intervention

    The key distinction from simple prediction is that we use the
    INFERRED exogenous variables specific to this student/context,
    not population averages.

    Usage:
        engine = CounterfactualEngine(scm, simulator, td_bkt)

        query = CounterfactualQuery(
            observed_outcome=failure,
            intervention={"Delta_t": +10},  # 10 more hours of study
            question="What if I had studied longer?"
        )

        result = engine.compute_counterfactual(query, belief_state)
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        simulator: Optional[StudentSimulator] = None,
        td_bkt: Optional[TemporalDifferenceBKT] = None,
        num_monte_carlo_samples: int = 100,
    ):
        """
        Initialize counterfactual engine.

        Args:
            scm: Structural Causal Model for causal reasoning
            simulator: Student simulator for trajectory Monte Carlo
            td_bkt: TD-BKT for state estimation
            num_monte_carlo_samples: Number of samples for Monte Carlo estimation
        """
        self.scm = scm
        self.td_bkt = td_bkt or scm.td_bkt
        self.simulator = simulator or StudentSimulator(StudentSimulatorConfig())
        self.num_samples = num_monte_carlo_samples

        logger.info("CounterfactualEngine initialized")

    def abduction(
        self,
        observed: EndogenousVariables,
        belief_state: BeliefState,
    ) -> ExogenousVariables:
        """
        Step 1: Abduction - Infer exogenous variables from observation.

        Given the observed outcome (e.g., failed quiz), we infer the
        most likely values of the background factors U = (U_student, U_content, U_noise).

        This uses Bayesian inference:
            P(U | Obs) ∝ P(Obs | U) * P(U)

        We use maximum likelihood estimation (MLE) to find the U that
        best explains the observed outcome.

        Args:
            observed: Observed endogenous variables including outcome
            belief_state: Current belief state for context

        Returns:
            Inferred exogenous variables
        """
        logger.debug(f"Abduction: inferring U from observed L_t={observed.L_t}, Obs={observed.Obs_t}")

        # Get prior mastery and context
        if observed.concept_id and observed.concept_id in belief_state.concept_states:
            concept_state = belief_state.concept_states[observed.concept_id]
            L_prior = concept_state.mastery_probability
        else:
            L_prior = 0.1

        # Define objective: find U that minimizes reconstruction error
        def objective(u_vector: np.ndarray) -> float:
            """Negative log-likelihood of observation given U"""
            u_student, u_content, u_noise = u_vector

            exo = ExogenousVariables(
                u_student=np.clip(u_student, -1, 1),
                u_content=np.clip(u_content, -1, 1),
                u_noise=np.clip(u_noise, -1, 1),
            )

            # Build state
            state = {
                "L_t_prev": L_prior,
                "A_t": observed.concept_id,
                "Delta_t": observed.Delta_t,
                "action_type": observed.action_type.value,
                "half_life_hours": observed.half_life_hours,
            }

            # Compute endogenous under this U
            endo = self.scm.compute_endogenous(state, exo)

            # Compute likelihood of observation
            p_correct, p_slip = self.scm.equations.observation_likelihood(
                L_t=endo.L_t,
                u_content=exo.u_content,
                u_noise=exo.u_noise,
            )

            if observed.Obs_t:
                p_obs = p_correct
            else:
                p_obs = 1 - p_correct

            # Add prior penalty (regularization toward 0)
            prior_penalty = 0.5 * (u_student**2 + u_content**2 + u_noise**2)

            # Negative log-likelihood
            nll = -np.log(p_obs + 1e-10) + prior_penalty

            return nll

        # Optimize to find best U
        result = optimize.minimize(
            objective,
            x0=np.array([0.0, 0.0, 0.0]),
            method='L-BFGS-B',
            bounds=[(-1, 1), (-1, 1), (-1, 1)],
        )

        inferred_u = ExogenousVariables(
            u_student=np.clip(result.x[0], -1, 1),
            u_content=np.clip(result.x[1], -1, 1),
            u_noise=np.clip(result.x[2], -1, 1),
        )

        logger.debug(f"Abduction complete: U_student={inferred_u.u_student:.3f}, "
                    f"U_content={inferred_u.u_content:.3f}, U_noise={inferred_u.u_noise:.3f}")

        return inferred_u

    def action(
        self,
        intervention: Dict[str, Any],
        current_scm_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Step 2: Action - Apply do-calculus intervention.

        This modifies the SCM state by fixing the intervened variable
        and removing its causal parents (breaking the causal link).

        Examples:
            - do(Delta_t = original + 10): Study 10 more hours
            - do(action_type = "review"): Review instead of practice
            - do(L_t_prev = 0.7): If prior mastery had been 0.7

        Args:
            intervention: Dict of {variable: new_value}
            current_scm_state: Current state before intervention

        Returns:
            Modified state after intervention
        """
        logger.debug(f"Action: applying intervention {intervention}")

        new_state = current_scm_state.copy()

        for var, value in intervention.items():
            if var == "Delta_t" and isinstance(value, (int, float)):
                # Additive intervention: add to current value
                if value >= 0:
                    new_state["Delta_t"] = current_scm_state.get("Delta_t", 0) + value
                else:
                    new_state["Delta_t"] = max(0, current_scm_state.get("Delta_t", 0) + value)
            elif var == "action_type":
                new_state["action_type"] = value
            else:
                # Direct assignment
                new_state[var] = value

            # Mark as intervened
            new_state[f"{var}_intervened"] = True

        return new_state

    def prediction(
        self,
        modified_scm_state: Dict[str, Any],
        inferred_exogenous: ExogenousVariables,
    ) -> EndogenousVariables:
        """
        Step 3: Prediction - Simulate counterfactual outcome.

        This computes what WOULD have happened under the intervention,
        using the INFERRED exogenous variables (not population average).

        This is the key distinction from standard prediction:
        - Standard: uses population U distribution
        - Counterfactual: uses U inferred from THIS specific observation

        Args:
            modified_scm_state: State after intervention
            inferred_exogenous: U inferred from actual observation

        Returns:
            Counterfactual endogenous variables
        """
        logger.debug("Prediction: computing counterfactual outcome")

        # Compute endogenous under intervention with inferred U
        cf_endo = self.scm.compute_endogenous(modified_scm_state, inferred_exogenous)

        # Copy context from state
        cf_endo.concept_id = modified_scm_state.get("concept_id")
        cf_endo.A_t = modified_scm_state.get("A_t")
        cf_endo.Delta_t = modified_scm_state.get("Delta_t", 0)

        return cf_endo

    def compute_counterfactual(
        self,
        query: CounterfactualQuery,
        belief_state: BeliefState,
    ) -> CounterfactualResult:
        """
        Complete counterfactual computation pipeline.

        Combines the 3 steps:
        1. Abduction: infer U from observed failure
        2. Action: apply intervention
        3. Prediction: simulate with inferred U

        Args:
            query: Counterfactual query with observation and intervention
            belief_state: Current belief state

        Returns:
            Counterfactual result with original vs counterfactual comparison
        """
        logger.info(f"Computing counterfactual: {query.question}")

        # Step 1: Abduction - infer U from observation
        inferred_u = self.abduction(query.observed_outcome, belief_state)

        # Step 2: Action - apply intervention
        original_state = self.scm.state_from_belief(
            belief_state,
            query.observed_outcome.concept_id or query.concept_id or "",
        )
        original_state["action_type"] = query.observed_outcome.action_type.value
        original_state["Delta_t"] = query.observed_outcome.Delta_t

        intervened_state = self.action(query.intervention, original_state)

        # Step 3: Prediction - compute counterfactual outcome
        cf_outcome = self.prediction(intervened_state, inferred_u)

        # Compute probability change
        prob_change = cf_outcome.L_t - query.observed_outcome.L_t

        # Compute confidence based on:
        # - How extreme the inferred U values are (less extreme = higher confidence)
        # - Consistency of the model fit
        u_extremity = (abs(inferred_u.u_student) + abs(inferred_u.u_content) + abs(inferred_u.u_noise)) / 3
        confidence = 1.0 - u_extremity * 0.5  # Scale to [0.5, 1.0]

        # Generate explanation
        explanation = self._generate_explanation(query, cf_outcome, prob_change)

        # Find alternative interventions that could help
        alternatives = self._find_alternative_interventions(
            query.observed_outcome,
            belief_state,
            inferred_u,
        )

        result = CounterfactualResult(
            original_outcome=query.observed_outcome,
            counterfactual_outcome=cf_outcome,
            probability_change=prob_change,
            inferred_exogenous=inferred_u,
            explanation=explanation,
            confidence=confidence,
            alternative_interventions=alternatives,
        )

        logger.info(f"Counterfactual complete: Δmastery={prob_change:+.2%}, confidence={confidence:.2f}")

        return result

    def _generate_explanation(
        self,
        query: CounterfactualQuery,
        cf_outcome: EndogenousVariables,
        prob_change: float,
    ) -> str:
        """Generate human-readable explanation of counterfactual"""
        direction = "increase" if prob_change > 0 else "decrease"
        magnitude = abs(prob_change)

        if magnitude < 0.05:
            impact = "minimal"
        elif magnitude < 0.15:
            impact = "moderate"
        else:
            impact = "significant"

        # Intervention-specific explanation
        if "Delta_t" in query.intervention:
            delta = query.intervention["Delta_t"]
            time_str = f"{delta:.0f} hours" if delta >= 1 else f"{delta*60:.0f} minutes"
            action_desc = f"studying {time_str} more" if delta > 0 else f"starting {abs(delta):.0f} hours earlier"
        elif "action_type" in query.intervention:
            action_desc = f"choosing {query.intervention['action_type']} instead"
        else:
            action_desc = "making this change"

        explanation = (
            f"Our analysis suggests that {action_desc} would have resulted in a "
            f"{impact} {direction} in your mastery ({prob_change:+.1%}). "
            f"Your estimated mastery would have been {cf_outcome.L_t:.1%} instead of "
            f"{query.observed_outcome.L_t:.1%}."
        )

        return explanation

    def _find_alternative_interventions(
        self,
        observed: EndogenousVariables,
        belief_state: BeliefState,
        inferred_u: ExogenousVariables,
    ) -> List[Dict[str, Any]]:
        """Find alternative interventions that could improve outcome"""
        alternatives = []

        # Test various interventions
        candidate_interventions = [
            {"Delta_t": 0.5, "description": "Study 30 more minutes"},
            {"Delta_t": 1.0, "description": "Study 1 more hour"},
            {"Delta_t": 2.0, "description": "Study 2 more hours"},
            {"action_type": ActionType.REVIEW.value, "description": "Review material instead"},
            {"action_type": ActionType.PREREQUISITE.value, "description": "Review prerequisites first"},
        ]

        original_state = self.scm.state_from_belief(
            belief_state,
            observed.concept_id or "",
        )
        original_state["action_type"] = observed.action_type.value
        original_state["Delta_t"] = observed.Delta_t

        for intervention in candidate_interventions:
            desc = intervention.pop("description", "")
            intervened_state = self.action(intervention, original_state)
            cf_outcome = self.prediction(intervened_state, inferred_u)

            improvement = cf_outcome.L_t - observed.L_t

            if improvement > 0.01:  # Only include positive improvements
                alternatives.append({
                    "intervention": intervention,
                    "description": desc,
                    "improvement": improvement,
                    "new_mastery": cf_outcome.L_t,
                })

        # Sort by improvement
        alternatives.sort(key=lambda x: x["improvement"], reverse=True)

        return alternatives[:5]  # Top 5

    def simulate_alternative_trajectory(
        self,
        belief_state: BeliefState,
        original_actions: List[Tuple[str, datetime, bool]],
        alternative_actions: List[Tuple[str, datetime, ActionType]],
        inferred_exogenous: Optional[ExogenousVariables] = None,
    ) -> TrajectoryComparison:
        """
        Monte Carlo simulation of alternative learning trajectories.

        Uses StudentSimulator with inferred exogenous variables to
        compare what actually happened vs what would have happened.

        Args:
            belief_state: Initial belief state
            original_actions: Actual (concept_id, timestamp, correct) sequence
            alternative_actions: Counterfactual (concept_id, timestamp, action_type) sequence
            inferred_exogenous: Optional fixed exogenous (samples if None)

        Returns:
            TrajectoryComparison with both trajectories
        """
        logger.info("Simulating alternative trajectory")

        # Infer exogenous from original trajectory if not provided
        if inferred_exogenous is None and original_actions:
            # Use last observation for abduction
            last_action = original_actions[-1]
            observed = EndogenousVariables(
                L_t=belief_state.get_concept_mastery(last_action[0]),
                Obs_t=last_action[2],
                concept_id=last_action[0],
                timestamp=last_action[1],
            )
            inferred_exogenous = self.abduction(observed, belief_state)

        if inferred_exogenous is None:
            inferred_exogenous = ExogenousVariables.sample_prior()

        # Simulate original trajectory
        original_trajectory = self.scm.forward_simulate(
            belief_state,
            [
                (a[0], a[1], ActionType.PRACTICE)
                for a in original_actions
            ],
            exogenous=inferred_exogenous,
            simulate_observations=False,
        )

        # Override with actual observations
        for i, (orig, action) in enumerate(zip(original_trajectory, original_actions)):
            orig.Obs_t = action[2]

        # Simulate counterfactual trajectory
        cf_trajectory = self.scm.forward_simulate(
            belief_state,
            alternative_actions,
            exogenous=inferred_exogenous,
            simulate_observations=True,
        )

        # Compute final masteries
        original_final = original_trajectory[-1].L_t if original_trajectory else 0.1
        cf_final = cf_trajectory[-1].L_t if cf_trajectory else 0.1

        # Compute 30-day retention
        # Simple exponential decay from final mastery
        half_life_orig = original_trajectory[-1].half_life_hours if original_trajectory else 24
        half_life_cf = cf_trajectory[-1].half_life_hours if cf_trajectory else 24

        day30_hours = 30 * 24
        retention_orig = original_final * math.pow(2, -day30_hours / half_life_orig)
        retention_cf = cf_final * math.pow(2, -day30_hours / half_life_cf)

        # Find divergence point
        divergence_point = None
        for i, (orig, cf) in enumerate(zip(original_trajectory, cf_trajectory)):
            if abs(orig.L_t - cf.L_t) > 0.05:  # 5% divergence threshold
                divergence_point = i
                break

        return TrajectoryComparison(
            original_trajectory=original_trajectory,
            counterfactual_trajectory=cf_trajectory,
            final_mastery_original=original_final,
            final_mastery_counterfactual=cf_final,
            improvement=cf_final - original_final,
            day30_retention_original=retention_orig,
            day30_retention_counterfactual=retention_cf,
            critical_divergence_point=divergence_point,
        )

    def compare_learning_paths(
        self,
        belief_state: BeliefState,
        path_a: List[Tuple[str, datetime]],
        path_b: List[Tuple[str, datetime]],
        num_simulations: int = 50,
    ) -> Dict[str, Any]:
        """
        Compare two alternative learning paths using Monte Carlo simulation.

        Args:
            belief_state: Initial belief state
            path_a: First path as list of (concept_id, timestamp)
            path_b: Second path as list of (concept_id, timestamp)
            num_simulations: Number of Monte Carlo samples

        Returns:
            Comparison statistics including expected outcomes and confidence intervals
        """
        logger.info(f"Comparing paths: {len(path_a)} vs {len(path_b)} steps")

        path_a_results = []
        path_b_results = []

        for _ in range(num_simulations):
            # Sample exogenous
            exo = ExogenousVariables.sample_prior()

            # Simulate path A
            traj_a = self.scm.forward_simulate(
                belief_state,
                [(cid, ts, ActionType.PRACTICE) for cid, ts in path_a],
                exogenous=exo,
            )

            # Simulate path B
            traj_b = self.scm.forward_simulate(
                belief_state,
                [(cid, ts, ActionType.PRACTICE) for cid, ts in path_b],
                exogenous=exo,
            )

            # Record final masteries
            if traj_a:
                path_a_results.append(traj_a[-1].L_t)
            if traj_b:
                path_b_results.append(traj_b[-1].L_t)

        # Compute statistics
        path_a_mean = np.mean(path_a_results) if path_a_results else 0
        path_b_mean = np.mean(path_b_results) if path_b_results else 0

        path_a_ci = (
            np.percentile(path_a_results, 5),
            np.percentile(path_a_results, 95),
        ) if path_a_results else (0, 0)

        path_b_ci = (
            np.percentile(path_b_results, 5),
            np.percentile(path_b_results, 95),
        ) if path_b_results else (0, 0)

        # Probability that B is better than A
        if path_a_results and path_b_results:
            prob_b_better = np.mean([b > a for a, b in zip(path_a_results, path_b_results)])
        else:
            prob_b_better = 0.5

        difference = path_b_mean - path_a_mean
        recommendation = "Path B" if difference > 0.02 else "Path A" if difference < -0.02 else "Either path"

        return {
            "path_a": {
                "mean_mastery": float(path_a_mean),
                "confidence_interval_95": [float(path_a_ci[0]), float(path_a_ci[1])],
                "num_steps": len(path_a),
            },
            "path_b": {
                "mean_mastery": float(path_b_mean),
                "confidence_interval_95": [float(path_b_ci[0]), float(path_b_ci[1])],
                "num_steps": len(path_b),
            },
            "difference": float(difference),
            "probability_b_better": float(prob_b_better),
            "recommendation": recommendation,
            "num_simulations": num_simulations,
        }
