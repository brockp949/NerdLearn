"""
Counterfactual Explanations API Endpoints

Research alignment:
- Pearl's Ladder of Causation (Association, Intervention, Counterfactual)
- Algorithmic Recourse with effort-aware constraints
- SHAP attribution for bandit decisions
- Socratic dialogue for pedagogical explanations

Endpoints:
- /explain: Generate counterfactual explanation
- /what-if: Evaluate multiple scenarios
- /compare-paths: Compare alternative learning paths
- /recourse/retrospective: What went wrong analysis
- /recourse/prospective: What to do next planning
- /recourse/time-constrained: Planning with time budget
- /explain-recommendation: SHAP explanation for recommendations
- /socratic/start: Start Socratic dialogue
- /socratic/continue: Continue dialogue
- /socratic/summary: Get dialogue summary
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

from app.schemas.counterfactual import (
    # Counterfactual
    CounterfactualExplanationRequest,
    CounterfactualExplanationResponse,
    CounterfactualResultResponse,
    RecoursePlanResponse,
    RecourseActionResponse,
    # Retrospective
    RetrospectiveAnalysisRequest,
    RetrospectiveAnalysisResponse,
    CriticalDecisionResponse,
    SuggestedCounterfactual,
    # Prospective
    ProspectivePlanRequest,
    ProspectivePlanResponse,
    SHAPFactorResponse,
    # Recommendation
    ExplainRecommendationRequest,
    ExplainRecommendationResponse,
    AlternativeArmResponse,
    # Path Comparison
    ComparePathsRequest,
    ComparePathsResponse,
    PathOutcomeResponse,
    SequenceContributionResponse,
    # What-If
    WhatIfRequest,
    WhatIfResponse,
    ScenarioOutcomeResponse,
    # Time-Constrained
    TimeConstrainedRecourseRequest,
    TimeConstrainedRecourseResponse,
    # Socratic
    StartDialogueRequest,
    StartDialogueResponse,
    ContinueDialogueRequest,
    ContinueDialogueResponse,
    DialogueSummaryResponse,
    DialogueTurnResponse,
    DialogueModeEnum,
)
from app.services.counterfactual_service import (
    CounterfactualService,
    ExplanationContext,
    get_counterfactual_service,
)
from app.adaptive.counterfactual import DialogueMode
from app.adaptive.td_bkt import BeliefState

logger = logging.getLogger(__name__)

router = APIRouter()


# Dependency injection
def get_service() -> CounterfactualService:
    """Get counterfactual service instance."""
    return get_counterfactual_service()


def _build_context(request) -> ExplanationContext:
    """Build ExplanationContext from request data."""
    belief_state = BeliefState(
        mastery=request.belief_state.mastery,
        uncertainty=request.belief_state.uncertainty or 0.1,
    )

    return ExplanationContext(
        student_id=request.student_id,
        concept_id=request.concept_id,
        concept_name=request.concept_name,
        belief_state=belief_state,
        recent_attempts=[a.model_dump() for a in request.recent_attempts],
        study_sessions=[s.model_dump() for s in getattr(request, 'study_sessions', [])],
        target_mastery=getattr(request, 'target_mastery', 0.85),
    )


# ============== Counterfactual Explanation ==============

@router.post("/explain", response_model=CounterfactualExplanationResponse)
async def explain_counterfactual(
    request: CounterfactualExplanationRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Generate a counterfactual explanation.

    Computes "what-if" scenarios using Pearl's 3-step counterfactual process:
    1. Abduction: Infer latent factors from observed outcome
    2. Action: Apply the intervention (do-operator)
    3. Prediction: Compute counterfactual outcome

    Returns the counterfactual result along with SHAP explanation
    and recourse plan if applicable.
    """
    try:
        context = _build_context(request)

        result = await service.get_counterfactual_explanation(
            context=context,
            intervention=request.intervention,
            question=request.question,
        )

        return CounterfactualExplanationResponse(
            query=result.query,
            counterfactual_result=CounterfactualResultResponse(
                original_probability=result.counterfactual_result.original_probability,
                counterfactual_probability=result.counterfactual_result.counterfactual_probability,
                probability_change=result.counterfactual_result.probability_change,
                explanation=result.counterfactual_result.explanation,
            ),
            shap_explanation=result.shap_explanation.__dict__ if result.shap_explanation else None,
            recourse_plan=RecoursePlanResponse(
                actions=[
                    RecourseActionResponse(
                        action=a.action_description if hasattr(a, 'action_description') else str(a),
                        feature=a.feature if hasattr(a, 'feature') else None,
                        original_value=a.original_value if hasattr(a, 'original_value') else None,
                        target_value=a.target_value if hasattr(a, 'target_value') else None,
                        effort=a.effort if hasattr(a, 'effort') else 0,
                        expected_impact=a.expected_impact if hasattr(a, 'expected_impact') else 0,
                    )
                    for a in result.recourse_plan.actions
                ],
                total_effort=result.recourse_plan.total_effort,
                time_estimate_minutes=result.recourse_plan.time_estimate_minutes,
                expected_probability=result.recourse_plan.expected_probability,
            ) if result.recourse_plan else None,
            natural_language_explanation=result.natural_language_explanation,
        )

    except Exception as e:
        logger.error(f"Counterfactual explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/what-if", response_model=WhatIfResponse)
async def what_if_analysis(
    request: WhatIfRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Evaluate multiple what-if scenarios.

    Compares multiple interventions to find the most impactful one.
    """
    try:
        context = _build_context(request)

        scenarios = []
        best_scenario = None
        best_change = float('-inf')

        for intervention in request.scenarios:
            result = await service.get_counterfactual_explanation(
                context=context,
                intervention=intervention,
            )

            scenario = ScenarioOutcomeResponse(
                intervention=intervention,
                original_probability=result.counterfactual_result.original_probability,
                counterfactual_probability=result.counterfactual_result.counterfactual_probability,
                probability_change=result.counterfactual_result.probability_change,
            )
            scenarios.append(scenario)

            if result.counterfactual_result.probability_change > best_change:
                best_change = result.counterfactual_result.probability_change
                best_scenario = intervention

        recommendation = None
        if best_scenario and best_change > 0:
            recommendation = (
                f"The most impactful intervention is {best_scenario}, "
                f"which could improve your success probability by {best_change:.1%}."
            )

        return WhatIfResponse(
            concept_id=request.concept_id,
            concept_name=request.concept_name,
            current_mastery=request.belief_state.mastery,
            scenarios=scenarios,
            best_scenario=best_scenario,
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"What-if analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Recourse Planning ==============

@router.post("/recourse/retrospective", response_model=RetrospectiveAnalysisResponse)
async def retrospective_analysis(
    request: RetrospectiveAnalysisRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Analyze what went wrong in past learning attempts.

    Identifies critical decision points that led to poor outcomes
    and explains contributing factors using SHAP attribution.
    """
    try:
        context = _build_context(request)

        result = await service.get_retrospective_analysis(context=context)

        if result.get("status") == "no_failures":
            return RetrospectiveAnalysisResponse(
                status="no_failures",
                concept_id=request.concept_id,
                concept_name=request.concept_name,
                current_mastery=request.belief_state.mastery,
                message=result.get("message"),
            )

        return RetrospectiveAnalysisResponse(
            status=result.get("status", "analyzed"),
            concept_id=result.get("concept_id"),
            concept_name=result.get("concept_name"),
            current_mastery=result.get("current_mastery"),
            num_failures=result.get("num_failures"),
            critical_decisions=[
                CriticalDecisionResponse(**d)
                for d in result.get("critical_decisions", [])
            ],
            shap_explanation=result.get("shap_explanation"),
            suggested_counterfactuals=[
                SuggestedCounterfactual(**s)
                for s in result.get("suggested_counterfactuals", [])
            ],
        )

    except Exception as e:
        logger.error(f"Retrospective analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recourse/prospective", response_model=ProspectivePlanResponse)
async def prospective_planning(
    request: ProspectivePlanRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Plan what to do next to achieve target mastery.

    Computes optimal recourse actions with effort-aware constraints.
    """
    try:
        context = _build_context(request)

        result = await service.get_prospective_plan(
            context=context,
            time_budget_minutes=request.time_budget_minutes,
            target_probability=request.target_probability,
        )

        return ProspectivePlanResponse(
            status=result.get("status", "planned"),
            concept_id=result.get("concept_id"),
            concept_name=result.get("concept_name"),
            current_mastery=result.get("current_mastery"),
            target_probability=result.get("target_probability"),
            time_budget=result.get("time_budget"),
            recourse_plan=RecoursePlanResponse(
                actions=[
                    RecourseActionResponse(**a)
                    for a in result.get("recourse_plan", {}).get("actions", [])
                ],
                total_effort=result.get("recourse_plan", {}).get("total_effort", 0),
                time_estimate_minutes=result.get("recourse_plan", {}).get("time_estimate_minutes", 0),
                expected_probability=result.get("recourse_plan", {}).get("expected_probability", 0),
            ),
            shap_factors={
                "positive": [
                    SHAPFactorResponse(**f)
                    for f in result.get("shap_factors", {}).get("positive", [])
                ],
                "negative": [
                    SHAPFactorResponse(**f)
                    for f in result.get("shap_factors", {}).get("negative", [])
                ],
            },
        )

    except Exception as e:
        logger.error(f"Prospective planning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recourse/time-constrained", response_model=TimeConstrainedRecourseResponse)
async def time_constrained_recourse(
    request: TimeConstrainedRecourseRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Plan recourse actions within a time budget.

    Maximizes probability of success subject to time constraint.
    """
    try:
        context = _build_context(request)

        result = await service.get_prospective_plan(
            context=context,
            time_budget_minutes=request.time_budget_minutes,
            target_probability=request.target_probability,
        )

        recourse_plan = result.get("recourse_plan", {})
        expected_prob = recourse_plan.get("expected_probability", context.belief_state.mastery)
        achievable = expected_prob >= request.target_probability

        message = (
            f"Target probability of {request.target_probability:.1%} is achievable "
            f"within {request.time_budget_minutes} minutes."
        ) if achievable else (
            f"Target probability of {request.target_probability:.1%} cannot be fully achieved "
            f"within {request.time_budget_minutes} minutes. Maximum achievable: {expected_prob:.1%}."
        )

        return TimeConstrainedRecourseResponse(
            concept_id=request.concept_id,
            time_budget_minutes=request.time_budget_minutes,
            target_probability=request.target_probability,
            achievable=achievable,
            recourse_plan=RecoursePlanResponse(
                actions=[
                    RecourseActionResponse(**a)
                    for a in recourse_plan.get("actions", [])
                ],
                total_effort=recourse_plan.get("total_effort", 0),
                time_estimate_minutes=recourse_plan.get("time_estimate_minutes", 0),
                expected_probability=expected_prob,
            ) if recourse_plan.get("actions") else None,
            max_achievable_probability=expected_prob,
            time_to_target=recourse_plan.get("time_estimate_minutes") if achievable else None,
            message=message,
        )

    except Exception as e:
        logger.error(f"Time-constrained recourse failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Recommendation Explanation ==============

@router.post("/explain-recommendation", response_model=ExplainRecommendationResponse)
async def explain_recommendation(
    request: ExplainRecommendationRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Explain why a learning activity was recommended.

    Uses SHAP to attribute the recommendation to context features.
    """
    try:
        context = _build_context(request)

        result = await service.explain_recommendation(
            context=context,
            recommended_arm=request.recommended_arm,
            arm_values=request.arm_values,
        )

        return ExplainRecommendationResponse(
            recommended_arm=result.get("recommended_arm"),
            explanation=result.get("explanation", {}),
            natural_language=result.get("natural_language", ""),
            alternative_arms=[
                AlternativeArmResponse(**a)
                for a in result.get("alternative_arms", [])
            ],
        )

    except Exception as e:
        logger.error(f"Recommendation explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Path Comparison ==============

@router.post("/compare-paths", response_model=ComparePathsResponse)
async def compare_paths(
    request: ComparePathsRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Compare two alternative learning paths.

    Uses Monte Carlo simulation to estimate outcomes and
    OrdShap to explain sequence differences.
    """
    try:
        context = _build_context(request)

        result = await service.compare_paths(
            context=context,
            path_a=request.path_a,
            path_b=request.path_b,
            num_simulations=request.num_simulations,
        )

        return ComparePathsResponse(
            path_a=PathOutcomeResponse(**result.get("path_a", {})),
            path_b=PathOutcomeResponse(**result.get("path_b", {})),
            difference=result.get("difference", {}),
            sequence_analysis={
                "value_contributions": [
                    SequenceContributionResponse(**c)
                    for c in result.get("sequence_analysis", {}).get("value_contributions", [])
                ]
            },
            recommendation=result.get("recommendation", ""),
        )

    except Exception as e:
        logger.error(f"Path comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Socratic Dialogue ==============

@router.post("/socratic/start", response_model=StartDialogueResponse)
async def start_socratic_dialogue(
    request: StartDialogueRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Start a new Socratic dialogue session.

    Generates initial tutor question based on causal analysis.
    """
    try:
        context = _build_context(request)

        # Map enum
        mode_map = {
            DialogueModeEnum.RETROSPECTIVE: DialogueMode.RETROSPECTIVE,
            DialogueModeEnum.PROSPECTIVE: DialogueMode.PROSPECTIVE,
            DialogueModeEnum.EXPLORATORY: DialogueMode.EXPLORATORY,
            DialogueModeEnum.CHALLENGE: DialogueMode.CHALLENGE,
        }
        mode = mode_map.get(request.mode, DialogueMode.EXPLORATORY)

        result = await service.start_socratic_dialogue(
            context=context,
            mode=mode,
        )

        return StartDialogueResponse(
            session_id=result.get("session_id"),
            mode=result.get("mode"),
            concept_id=result.get("concept_id"),
            concept_name=result.get("concept_name"),
            current_mastery=result.get("current_mastery"),
            target_mastery=result.get("target_mastery"),
            turns=[
                DialogueTurnResponse(**t)
                for t in result.get("turns", [])
            ],
        )

    except Exception as e:
        logger.error(f"Start Socratic dialogue failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/socratic/continue", response_model=ContinueDialogueResponse)
async def continue_socratic_dialogue(
    request: ContinueDialogueRequest,
    service: CounterfactualService = Depends(get_service),
):
    """
    Continue an existing Socratic dialogue.

    Handles student response and generates next tutor question.
    """
    try:
        result = await service.continue_socratic_dialogue(
            session_id=request.session_id,
            student_message=request.student_message,
        )

        return ContinueDialogueResponse(
            session_id=result.get("session_id"),
            turns=[
                DialogueTurnResponse(**t)
                for t in result.get("turns", [])
            ],
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Continue Socratic dialogue failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/socratic/{session_id}/summary", response_model=DialogueSummaryResponse)
async def get_dialogue_summary(
    session_id: str,
    service: CounterfactualService = Depends(get_service),
):
    """
    Get a summary of a Socratic dialogue session.
    """
    try:
        result = await service.get_dialogue_summary(session_id=session_id)

        return DialogueSummaryResponse(
            session_id=result.get("session_id"),
            summary=result.get("summary", ""),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get dialogue summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
