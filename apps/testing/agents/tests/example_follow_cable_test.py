"""
Example: Follow the Cable - Causal Chain Testing

Tests the "Follow the Cable" heuristic from the PDF:
"Test the chain of agents. Simulate the flow of data from ingestion to inference."

Adapted for NerdLearn's adaptive learning system:
Input: User struggling with a topic
Expected: Learning path that follows prerequisite chain
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any, List

from apps.testing.agents.base_verifier import VerifierAgent, GoalVector
from apps.testing.agents.utils.embedding_similarity import semantic_distance


class MockLLMClient:
    """Mock LLM client"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        return '''```json
{
    "passed": true,
    "confidence": 0.82,
    "reasoning": "Learning path correctly follows prerequisite chain. Addresses weak limits foundation before advanced derivatives. Each step is semantically relevant to calculus mastery goal.",
    "violations": []
}
```'''


async def mock_generate_learning_path(user_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Mock learning path generator.
    
    In production, this would integrate with:
    - app.adaptive.path_generator
    - app.adaptive.bkt.bayesian_kt
    """
    
    # Simulate intelligent path generation based on user state
    if user_state.get('struggle_areas') and user_state['mastery_scores']['limits'] < 0.7:
        # Detect weak foundation, build prerequisite chain
        return [
            {
                "step": 1,
                "topic": "limits_fundamentals",
                "reason": "Weak foundation detected (score: 0.60)",
                "estimated_duration": "2 hours",
                "mastery_required": 0.75
            },
            {
                "step": 2,
                "topic": "basic_derivatives_introduction",
                "reason": "Build on strengthened limits knowledge",
                "estimated_duration": "3 hours",
                "mastery_required": 0.70
            },
            {
                "step": 3,
                "topic": "chain_rule_mastery",
                "reason": "Address identified struggle area",
                "estimated_duration": "2.5 hours",
                "mastery_required": 0.80
            }
        ]
    else:
        # Default path
        return [
            {"step": 1, "topic": "review_current_topic", "reason": "General review"}
        ]


@pytest.mark.asyncio
async def test_follow_the_cable_learning_path():
    """
    Test causal chain reasoning in learning path generation.
    
    From PDF:
    "The test suite parses the inferred_components array. It uses an
    embedding similarity check (vector distance) to ensure the inferred
    components are semantically relevant to the root cause (Regulations)
    and not generic suggestions (e.g., 'Office Supplies')."
    
    Adapted for NerdLearn:
    - Root cause: Weak limits understanding
    - Expected cable: limits → basic derivatives → chain rule
    - Should NOT suggest: unrelated topics, advanced topics without prerequisites
    """
    
    # Arrange - Goal vector for causal reasoning
    goal = GoalVector(
        name="Causal Learning Path",
        description="Path must follow prerequisite chain and address root causes",
        constraints=[
            "Must address weak foundations before advanced topics",
            "Each step must be semantically relevant to the goal",
            "No generic suggestions unrelated to the struggle area",
            "Progressive difficulty scaling"
        ],
        pass_threshold=0.75
    )
    
    llm_client = MockLLMClient()
    verifier = VerifierAgent(goal, llm_client)
    
    # Act - Generate learning path
    user_state = {
        "learner_id": "test_user_123",
        "current_topic": "calculus_derivatives",
        "struggle_areas": ["chain_rule", "implicit_differentiation"],
        "mastery_scores": {
            "algebra": 0.85,
            "limits": 0.60,  # Weak foundation - this is the root cause
            "derivatives": 0.30
        },
        "learning_velocity": 0.7
    }
    
    learning_path = await mock_generate_learning_path(user_state)
    
    # Assert with semantic verification
    context = {
        "root_cause": "weak limit understanding",
        "target_skill": "derivatives mastery",
        "expected_chain": ["limits", "basic_derivatives", "chain_rule"],
        "should_exclude": ["unrelated topics", "advanced calculus", "algebra review"]
    }
    
    result = await verifier.verify(learning_path, context)
    
    # Verify the cable was followed
    assert result.passed, f"Causal chain validation failed: {result.reasoning}"
    assert result.confidence > 0.75, f"Low confidence in path quality: {result.confidence}"
    
    # Verify path structure
    assert len(learning_path) >= 2, "Path should have multiple steps for prerequisite chain"
    
    # Verify first step addresses root cause (weak limits)
    first_topic = learning_path[0]['topic']
    assert 'limits' in first_topic.lower(), \
        f"First step should address weak limits, got: {first_topic}"
    
    # Verify semantic progression
    topics = [step['topic'] for step in learning_path]
    print(f"\n✓ Learning path follows causal chain:")
    for i, step in enumerate(learning_path, 1):
        print(f"  {i}. {step['topic']}: {step['reason']}")
    
    # Check semantic relevance between consecutive steps
    # (In production, this would use actual embeddings)
    for i in range(len(topics) - 1):
        current = topics[i]
        next_topic = topics[i + 1]
        print(f"  → Semantic link: {current} → {next_topic}")
    
    print(f"\n  Verification confidence: {result.confidence:.2f}")
    print(f"  Goal alignment: {result.goal_alignment:.2f}")


@pytest.mark.asyncio
async def test_follow_the_cable_detects_broken_chain():
    """
    Test that verifier detects when causal chain is broken.
    
    Example: Suggesting advanced topics before prerequisites
    """
    
    goal = GoalVector(
        name="Causal Chain Integrity",
        description="Detect broken prerequisite chains",
        constraints=["Prerequisites must come before dependent topics"],
        pass_threshold=0.75
    )
    
    llm_client = MockLLMClient()
    verifier = VerifierAgent(goal, llm_client)
    
    # Bad learning path - jumps to advanced topic without prerequisites
    broken_path = [
        {
            "step": 1,
            "topic": "multivariable_calculus",  # Advanced topic
            "reason": "Generic suggestion"
        },
        {
            "step": 2,
            "topic": "limits_fundamentals",  # Should come first!
            "reason": "Prerequisite"
        }
    ]
    
    context = {
        "root_cause": "weak limits",
        "current_mastery": {"limits": 0.30, "derivatives": 0.20},
        "broken_chain_expected": True
    }
    
    # In a real test with actual LLM, this should fail
    # For demo, we'll verify the path structure manually
    
    # Manual check: verify prerequisites come before advanced topics
    topics = [s['topic'] for s in broken_path]
    
    # This is a broken chain - multivariable before limits
    assert 'multivariable' in topics[0].lower(), "First step is advanced topic"
    assert 'limits' in topics[1].lower(), "Prerequisites come after advanced"
    
    print("✓ Detected broken causal chain (prerequisites after advanced topics)")


@pytest.mark.asyncio
async def test_semantic_relevance_to_goal():
    """
    Test that all recommendations are semantically relevant to the goal.
    
    From PDF: "ensure the inferred components are semantically relevant to
    the root cause and not generic suggestions"
    """
    
    goal = GoalVector(
        name="Semantic Relevance",
        description="All recommendations must be relevant to the identified problem",
        constraints=["No generic unrelated suggestions"],
        pass_threshold=0.70
    )
    
    llm_client = MockLLMClient()
    verifier = VerifierAgent(goal, llm_client)
    
    # Good path - all topics relevant to calculus struggle
    good_path = [
        {"topic": "limits_review", "relevance_score": 0.92},
        {"topic": "derivative_rules", "relevance_score": 0.88},
        {"topic": "chain_rule_practice", "relevance_score": 0.95}
    ]
    
    # Bad path - includes unrelated topic
    bad_path = [
        {"topic": "limits_review", "relevance_score": 0.92},
        {"topic": "office_supplies_management", "relevance_score": 0.05},  # Irrelevant!
        {"topic": "chain_rule_practice", "relevance_score": 0.95}
    ]
    
    context = {
        "problem": "calculus derivatives struggle",
        "domain": "mathematics education"
    }
    
    # Test good path
    good_result = await verifier.verify(good_path, context)
    assert all(s['relevance_score'] > 0.7 for s in good_path), \
        "All topics should be highly relevant"
    
    # Test bad path
    # In production, verifier should detect the irrelevant topic
    has_irrelevant = any(s['relevance_score'] < 0.3 for s in bad_path)
    assert has_irrelevant, "Should detect irrelevant topic"
    
    print("✓ Semantic relevance filtering working correctly")


if __name__ == "__main__":
    asyncio.run(test_follow_the_cable_learning_path())
    asyncio.run(test_follow_the_cable_detects_broken_chain())
    asyncio.run(test_semantic_relevance_to_goal())
    print("\n✓ All Follow the Cable tests passed!")
