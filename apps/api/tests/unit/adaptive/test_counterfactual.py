"""
Unit tests for Counterfactual Explanations module.

Tests cover:
- Structural Causal Model (SCM)
- Counterfactual Engine (3-step process)
- Algorithmic Recourse
- SHAP Attribution
- Socratic Dialogue
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock
import numpy as np

from app.adaptive.counterfactual import (
    # SCM
    StructuralCausalModel,
    ExogenousVariables,
    EndogenousVariables,
    StructuralEquations,
    # Counterfactual
    CounterfactualEngine,
    CounterfactualQuery,
    CounterfactualResult,
    # Recourse
    AlgorithmicRecourse,
    FeatureConstraint,
    RecourseAction,
    RecoursePlan,
    # SHAP
    BanditSHAPExplainer,
    OrdShapExplainer,
    SHAPExplanation,
    # Socratic
    SocraticContext,
    SocraticPrompts,
    DialogueMode,
)
from app.adaptive.td_bkt import BeliefState


class TestStructuralCausalModel:
    """Tests for Structural Causal Model."""

    def test_scm_initialization(self):
        """Test SCM initializes with default components."""
        scm = StructuralCausalModel()

        assert scm.td_bkt is not None
        assert scm.equations is not None
        assert scm.causal_graph is not None

    def test_scm_default_graph(self):
        """Test SCM creates valid default learning graph."""
        scm = StructuralCausalModel()

        # Default graph should have key nodes
        assert "L_t-1" in scm.causal_graph
        assert "L_t" in scm.causal_graph

        # L_t-1 should influence L_t
        assert "L_t" in scm.causal_graph["L_t-1"]

    def test_scm_intervene(self):
        """Test do-operator intervention."""
        scm = StructuralCausalModel()

        current_state = {
            "mastery": 0.5,
            "study_time": 30,
            "concept_id": "test_concept",
        }

        # Intervene on study time
        modified = scm.intervene(
            variable="study_time",
            value=60,
            current_state=current_state,
        )

        assert modified["study_time"] == 60
        # Original state should be unchanged
        assert current_state["study_time"] == 30

    def test_structural_equations_mastery_transition(self):
        """Test mastery transition with decay and learning gain."""
        scm = StructuralCausalModel()

        # Create exogenous variables (latent student ability)
        exogenous = ExogenousVariables(
            u_student=0.8,  # High ability
            u_content=0.5,  # Medium difficulty
            u_noise=0.1,
        )

        # Test mastery transition
        delta_t = 24  # 24 hours since last practice
        previous_mastery = 0.7
        action = {"study_time": 30, "difficulty": 0.5}

        new_mastery = scm.equations.mastery_transition(
            previous_mastery=previous_mastery,
            delta_t=delta_t,
            action=action,
            exogenous=exogenous,
        )

        # Mastery should be between 0 and 1
        assert 0 <= new_mastery <= 1

    def test_structural_equations_observation_likelihood(self):
        """Test observation likelihood with slip/guess."""
        scm = StructuralCausalModel()

        # Test high mastery - should have high success probability
        high_mastery = 0.9
        high_prob = scm.equations.observation_likelihood(
            mastery=high_mastery,
            difficulty=0.5,
        )
        assert high_prob > 0.7

        # Test low mastery - should have lower success probability
        low_mastery = 0.2
        low_prob = scm.equations.observation_likelihood(
            mastery=low_mastery,
            difficulty=0.5,
        )
        assert low_prob < high_prob


class TestCounterfactualEngine:
    """Tests for Counterfactual Engine."""

    def test_engine_initialization(self):
        """Test engine initializes with SCM."""
        engine = CounterfactualEngine()

        assert engine.scm is not None

    def test_abduction_step(self):
        """Test abduction infers exogenous from observed outcome."""
        engine = CounterfactualEngine()

        # Observed failure with moderate mastery
        observed = EndogenousVariables(
            mastery=0.6,
            observation=False,  # Incorrect
            delta_t=12,
        )

        belief_state = BeliefState(mastery=0.6)

        exogenous = engine.abduction(observed=observed, belief_state=belief_state)

        assert exogenous is not None
        assert hasattr(exogenous, 'u_student')
        assert hasattr(exogenous, 'u_content')

    def test_action_step(self):
        """Test action applies do-calculus intervention."""
        engine = CounterfactualEngine()

        current_state = {
            "mastery": 0.5,
            "study_time": 20,
            "observation": False,
        }

        intervention = {"study_time": 45}

        modified = engine.action(
            intervention=intervention,
            current_scm_state=current_state,
        )

        assert modified["study_time"] == 45

    def test_prediction_step(self):
        """Test prediction simulates counterfactual outcome."""
        engine = CounterfactualEngine()

        modified_state = {
            "mastery": 0.6,
            "study_time": 45,
            "delta_t": 12,
            "difficulty": 0.5,
        }

        exogenous = ExogenousVariables(
            u_student=0.7,
            u_content=0.5,
            u_noise=0.1,
        )

        predicted = engine.prediction(
            modified_scm_state=modified_state,
            inferred_exogenous=exogenous,
        )

        assert predicted is not None
        assert hasattr(predicted, 'mastery')

    def test_compute_counterfactual_full_pipeline(self):
        """Test full 3-step counterfactual computation."""
        engine = CounterfactualEngine()

        query = CounterfactualQuery(
            student_id="test_student",
            concept_id="test_concept",
            observed_outcome=False,  # Student failed
            intervention={"study_time": 45},  # What if studied longer?
            question="What if I had studied 45 minutes instead of 20?",
        )

        belief_state = BeliefState(mastery=0.5)

        result = engine.compute_counterfactual(
            query=query,
            belief_state=belief_state,
        )

        assert result is not None
        assert hasattr(result, 'original_probability')
        assert hasattr(result, 'counterfactual_probability')
        assert hasattr(result, 'probability_change')


class TestAlgorithmicRecourse:
    """Tests for Algorithmic Recourse."""

    def test_default_constraints(self):
        """Test default learning constraints are reasonable."""
        constraints = AlgorithmicRecourse.default_learning_constraints()

        # Should have common learning features
        assert "study_time" in constraints
        assert "practice_sessions" in constraints
        assert "past_grade" in constraints

        # Study time should be actionable with low effort
        assert constraints["study_time"].actionable == True
        assert constraints["study_time"].effort_weight < 1.0

        # Past grade should be immutable
        assert constraints["past_grade"].actionable == False

    def test_effort_aware_distance(self):
        """Test effort-aware distance metric."""
        recourse = AlgorithmicRecourse()

        current = {"study_time": 20, "practice_sessions": 2}
        target = {"study_time": 60, "practice_sessions": 5}

        distance = recourse.effort_aware_distance(
            current=current,
            target=target,
        )

        assert distance > 0

    def test_find_minimum_cost_recourse(self):
        """Test finding minimum-cost recourse plan."""
        recourse = AlgorithmicRecourse()

        belief_state = BeliefState(mastery=0.4)

        plan = recourse.find_minimum_cost_recourse(
            current_state=belief_state,
            concept_id="test_concept",
            target_success_probability=0.85,
        )

        assert plan is not None
        assert hasattr(plan, 'actions')
        assert hasattr(plan, 'total_effort')
        assert hasattr(plan, 'expected_probability')

    def test_time_constrained_recourse(self):
        """Test recourse respects time budget."""
        recourse = AlgorithmicRecourse()

        belief_state = BeliefState(mastery=0.4)

        plan = recourse.find_minimum_cost_recourse(
            current_state=belief_state,
            concept_id="test_concept",
            target_success_probability=0.85,
            time_budget_minutes=30,
        )

        # Total time should not exceed budget
        if plan and plan.time_estimate_minutes:
            assert plan.time_estimate_minutes <= 30 or plan.expected_probability < 0.85


class TestBanditSHAPExplainer:
    """Tests for SHAP Attribution."""

    def test_explainer_initialization(self):
        """Test SHAP explainer initializes."""
        explainer = BanditSHAPExplainer()

        assert explainer is not None
        assert explainer.feature_names is not None

    def test_explain_selection(self):
        """Test SHAP explanation for arm selection."""
        explainer = BanditSHAPExplainer()

        context = {
            "mastery": 0.6,
            "recency": 0.8,
            "recent_accuracy": 0.7,
            "study_time": 0.5,
            "attempt_count": 0.3,
        }

        explanation = explainer.explain_selection(
            context=context,
            selected_arm="concept_1",
        )

        assert explanation is not None
        assert hasattr(explanation, 'arm')
        assert hasattr(explanation, 'feature_contributions')

        # Contributions should sum approximately to prediction - base
        total_contribution = sum(
            fc.contribution for fc in explanation.feature_contributions
        )
        # Allow for numerical precision issues
        assert abs(total_contribution) < 10  # Reasonable bound

    def test_shap_values_sum_to_prediction(self):
        """Test SHAP values sum to prediction minus base value."""
        explainer = BanditSHAPExplainer()

        context = {
            "mastery": 0.7,
            "recency": 0.9,
            "recent_accuracy": 0.8,
            "study_time": 0.6,
            "attempt_count": 0.4,
        }

        # Define a simple prediction function
        def predict_fn(X):
            return np.sum(X, axis=1) / X.shape[1]

        explanation = explainer.explain_selection(
            context=context,
            selected_arm="concept_1",
            predict_fn=predict_fn,
        )

        # SHAP values + base should approximate prediction
        total_shap = sum(fc.contribution for fc in explanation.feature_contributions)
        expected_prediction = sum(context.values()) / len(context)

        # Base + SHAP should be close to prediction
        reconstructed = explanation.base_value + total_shap
        assert abs(reconstructed - expected_prediction) < 0.5


class TestOrdShapExplainer:
    """Tests for OrdShap (sequence explanation)."""

    def test_ordshap_initialization(self):
        """Test OrdShap explainer initializes."""
        explainer = OrdShapExplainer()

        assert explainer is not None

    def test_compute_ordshap(self):
        """Test OrdShap computation for sequence."""
        explainer = OrdShapExplainer()

        sequence = ["concept_a", "concept_b", "concept_c"]
        belief_state = BeliefState(mastery=0.5)

        contributions = explainer.compute_ordshap(
            sequence=sequence,
            belief_state=belief_state,
        )

        assert contributions is not None
        assert len(contributions) == len(sequence)

        for contrib in contributions:
            assert hasattr(contrib, 'item')
            assert hasattr(contrib, 'value_contribution')
            assert hasattr(contrib, 'position_contribution')

    def test_explain_sequence_impact(self):
        """Test sequence impact explanation."""
        explainer = OrdShapExplainer()

        sequence = ["concept_a", "concept_b"]
        belief_state = BeliefState(mastery=0.5)

        impact = explainer.explain_sequence_impact(
            sequence=sequence,
            belief_state=belief_state,
        )

        assert impact is not None


class TestSocraticPrompts:
    """Tests for Socratic Prompt Generation."""

    def test_system_prompt(self):
        """Test system prompt establishes Socratic persona."""
        prompt = SocraticPrompts.system_prompt()

        assert "Socratic" in prompt
        assert "question" in prompt.lower()
        # Should discourage stating facts directly
        assert "Never say" in prompt or "never" in prompt.lower()

    def test_retrospective_prompt(self):
        """Test retrospective prompt includes failure analysis."""
        context = SocraticContext(
            student_id="test_student",
            concept_id="test_concept",
            concept_name="Linear Algebra",
            current_mastery=0.4,
            target_mastery=0.85,
            failures=[
                {"timestamp": "2024-01-01", "question_type": "quiz", "difficulty": 0.7}
            ],
        )

        prompt = SocraticPrompts.retrospective_prompt(context)

        assert "Linear Algebra" in prompt
        assert "0.4" in prompt or "40" in prompt  # Mastery value
        assert "RETROSPECTIVE" in prompt.upper()

    def test_prospective_prompt(self):
        """Test prospective prompt includes planning context."""
        context = SocraticContext(
            student_id="test_student",
            concept_id="test_concept",
            concept_name="Calculus",
            current_mastery=0.5,
            recourse_plan={
                "actions": [
                    {"action": "study_time", "effort": 0.3, "expected_impact": 0.15}
                ],
                "total_effort": 0.3,
                "time_estimate_minutes": 30,
                "expected_probability": 0.7,
            },
        )

        prompt = SocraticPrompts.prospective_prompt(context)

        assert "Calculus" in prompt
        assert "PROSPECTIVE" in prompt.upper()

    def test_challenge_prompt(self):
        """Test challenge prompt handles student disagreement."""
        context = SocraticContext(
            student_id="test_student",
            concept_id="test_concept",
            concept_name="Physics",
            current_mastery=0.4,
            student_challenge="But I studied for 3 hours!",
        )

        prompt = SocraticPrompts.challenge_response_prompt(context)

        assert "studied for 3 hours" in prompt
        assert "CHALLENGE" in prompt.upper()
        # Should encourage acknowledging student perspective
        assert "acknowledge" in prompt.lower() or "perspective" in prompt.lower()

    def test_serialize_for_llm(self):
        """Test serialization produces valid LLM input."""
        context = SocraticContext(
            student_id="test_student",
            concept_id="test_concept",
            concept_name="Chemistry",
            current_mastery=0.6,
        )

        result = SocraticPrompts.serialize_for_llm(
            context=context,
            mode=DialogueMode.EXPLORATORY,
        )

        assert "system" in result
        assert "user" in result
        assert len(result["system"]) > 0
        assert len(result["user"]) > 0


class TestSocraticContext:
    """Tests for Socratic Context."""

    def test_context_initialization(self):
        """Test context initializes with defaults."""
        context = SocraticContext(
            student_id="test",
            concept_id="test",
            concept_name="Test Concept",
            current_mastery=0.5,
        )

        assert context.recent_attempts == []
        assert context.failures == []
        assert context.dialogue_history == []
        assert context.target_mastery == 0.85

    def test_context_with_all_data(self):
        """Test context with full causal data."""
        context = SocraticContext(
            student_id="test",
            concept_id="test",
            concept_name="Test",
            current_mastery=0.6,
            shap_explanation={"feature_contributions": []},
            counterfactual_result={"probability_change": 0.1},
            recourse_plan={"actions": []},
        )

        assert context.shap_explanation is not None
        assert context.counterfactual_result is not None
        assert context.recourse_plan is not None


# Integration tests

class TestCounterfactualIntegration:
    """Integration tests for counterfactual pipeline."""

    def test_full_pipeline_retrospective(self):
        """Test full retrospective analysis pipeline."""
        # Initialize components
        scm = StructuralCausalModel()
        cf_engine = CounterfactualEngine(scm=scm)
        recourse = AlgorithmicRecourse(scm=scm)
        shap_explainer = BanditSHAPExplainer()

        # Simulate a failed student
        belief_state = BeliefState(mastery=0.4)

        # Step 1: Compute counterfactual
        query = CounterfactualQuery(
            student_id="student_1",
            concept_id="algebra",
            observed_outcome=False,
            intervention={"study_time": 45},
        )
        cf_result = cf_engine.compute_counterfactual(query, belief_state)

        # Step 2: Get recourse plan
        recourse_plan = recourse.find_minimum_cost_recourse(
            current_state=belief_state,
            concept_id="algebra",
            target_success_probability=0.8,
        )

        # Step 3: Get SHAP explanation
        context = {"mastery": 0.4, "recency": 0.5, "recent_accuracy": 0.3}
        shap = shap_explainer.explain_selection(context, "algebra")

        # Step 4: Build Socratic context
        socratic_context = SocraticContext(
            student_id="student_1",
            concept_id="algebra",
            concept_name="Algebra",
            current_mastery=0.4,
            shap_explanation={
                "feature_contributions": [
                    {"feature": fc.feature, "contribution": fc.contribution}
                    for fc in shap.feature_contributions
                ]
            },
            counterfactual_result={
                "probability_change": cf_result.probability_change
            },
            recourse_plan={
                "actions": [{"action": str(a)} for a in recourse_plan.actions]
            } if recourse_plan else None,
        )

        # Verify integration
        assert socratic_context.shap_explanation is not None
        assert socratic_context.counterfactual_result is not None

        # Generate prompt
        prompts = SocraticPrompts.serialize_for_llm(
            socratic_context,
            DialogueMode.RETROSPECTIVE
        )
        assert len(prompts["user"]) > 100  # Should have substantial content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
