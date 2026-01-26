"""
Counterfactual Explanations for Adaptive Learning Paths

This module implements the Causal-Counterfactual Framework integrating:
- Structural Causal Models (SCM) for formalizing learning dynamics
- Counterfactual computation via Abduction-Action-Prediction
- Algorithmic recourse for actionable recommendations
- SHAP attribution for explaining bandit decisions
- Socratic dialogue generation from causal insights

Key Components:
- StructuralCausalModel: Wraps TD-BKT with formal causal semantics
- CounterfactualEngine: Computes "what-if" scenarios using Pearl's 3-step process
- AlgorithmicRecourse: Finds minimum-cost actions to improve outcomes
- BanditSHAPExplainer: Explains MAB arm selection with feature attribution
- SocraticAgent: Converts causal data into pedagogical dialogue

Research Basis:
- Pearl's Ladder of Causation (Association, Intervention, Counterfactual)
- Time-Dependent Bayesian Knowledge Tracing (TD-BKT)
- Shapley Additive Explanations (SHAP)
- Algorithmic Recourse with effort-aware constraints
"""

from app.adaptive.counterfactual.scm import (
    StructuralCausalModel,
    ExogenousVariables,
    EndogenousVariables,
    StructuralEquations,
)
from app.adaptive.counterfactual.counterfactual_engine import (
    CounterfactualEngine,
    CounterfactualQuery,
    CounterfactualResult,
)
from app.adaptive.counterfactual.recourse import (
    AlgorithmicRecourse,
    FeatureConstraint,
    RecourseAction,
    RecoursePlan,
)
from app.adaptive.counterfactual.shap_attribution import (
    BanditSHAPExplainer,
    OrdShapExplainer,
    SHAPExplanation,
)
from app.adaptive.counterfactual.socratic_prompts import (
    SocraticContext,
    SocraticPrompts,
    DialogueMode,
)
from app.adaptive.counterfactual.socratic_agent import (
    SocraticAgent,
    SocraticDialogue,
    SocraticDialogueBuilder,
    explain_failure_socratically,
    plan_next_steps_socratically,
)

__all__ = [
    # SCM
    "StructuralCausalModel",
    "ExogenousVariables",
    "EndogenousVariables",
    "StructuralEquations",
    # Counterfactual Engine
    "CounterfactualEngine",
    "CounterfactualQuery",
    "CounterfactualResult",
    # Recourse
    "AlgorithmicRecourse",
    "FeatureConstraint",
    "RecourseAction",
    "RecoursePlan",
    # SHAP
    "BanditSHAPExplainer",
    "OrdShapExplainer",
    "SHAPExplanation",
    # Socratic
    "SocraticContext",
    "SocraticPrompts",
    "DialogueMode",
    "SocraticAgent",
    "SocraticDialogue",
    "SocraticDialogueBuilder",
    "explain_failure_socratically",
    "plan_next_steps_socratically",
]
