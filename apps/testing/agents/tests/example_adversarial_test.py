"""
Example: Adversarial Testing - Chaos Monkey

Tests system robustness with poisoned data.

From PDF:
"Using the concepts from NerdLearn's 'Adversarial Peer,' we purposely
inject 'poisoned' data to test the system's robustness."
"""

import pytest
import asyncio
from typing import Dict, Any

from apps.testing.agents.base_verifier import VerifierAgent, GoalVector
from apps.testing.agents.adversarial_peer import AdversarialPeer, PoisonType


class MockLLMClient:
    """Mock LLM for testing"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        # Simulate detection of conflict
        if 'conflict' in prompt.lower():
            return '''```json
{
    "passed": true,
    "confidence": 0.75,
    "reasoning": "System correctly detected relationship conflict and lowered confidence. Despite high skill match, negative relationship fit should prevent recommendation.",
    "violations": []
}
```'''
        return '''```json
{
    "passed": true,
    "confidence": 0.80,
    "reasoning": "Handled edge case appropriately.",
    "violations": []
}
```'''


async def mock_content_recommender(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock content recommendation system.
    
    Should detect conflicts and lower confidence appropriately.
    """
    
    # Check for relationship conflict poison
    if '_adversarial' in input_data:
        adversarial = input_data['_adversarial']
        if adversarial['type'] == 'relationship_conflict':
            # System should be cautious!
            return {
                "recommendation": "Introduce contacts",
                "confidence": 0.25,  # Low due to conflict
                "warnings": [f"Relationship conflict detected: {input_data.get('conflict_reason')}"],
                "should_proceed": False,
                "reasoning": "High skill match but incompatible relationship dynamics"
            }
    
    # Check for conflicting sources
    if 'sources' in input_data:
        sources = input_data['sources']
        if 'climate_denial' in str(sources).lower() and 'scientific_consensus' in str(sources).lower():
            return {
                "recommendation": "Review content",
                "confidence": 0.40,  # Lowered due to conflict
                "warnings": ["Conflicting information from sources"],
                "should_flag": True
            }
    
    # Default high confidence
    return {
        "recommendation": "Proceed",
        "confidence": 0.85,
        "warnings": []
    }


@pytest.mark.asyncio
async def test_adversarial_relationship_conflict():
    """
    Test from PDF:
    "The Adversarial Peer introduces two contacts with high influence
    but negative 'Relationship Fit' (e.g., they are competitors).
    System should realize that introducing two enemies will kill the
    project, even if their skills are perfect."
    
    Pass Condition: confidence score < 0.3
    """
    
    # Arrange
    goal = GoalVector(
        name="Conflict Detection",
        description="Detect and handle conflicting relationships appropriately",
        constraints=[
            "Lower confidence when conflicts detected",
            "Flag relationship incompatibilities",
            "Prioritize relationship fit over skill match"
        ],
        pass_threshold=0.70
    )
    
    llm_client = MockLLMClient()
    verifier = VerifierAgent(goal, llm_client)
    adversarial_peer = AdversarialPeer(llm_client)
    
    # Generate poisoned data
    base_data = {
        "contact_a": "Alice",
        "contact_b": "Bob",
        "project": "NerdLearn feature development"
    }
    
    poisoned_data = adversarial_peer.inject_relationship_conflict(
        base_data,
        conflict_reason="competitors with history of conflict"
    )
    
    print(f"\n🧪 Adversarial scenario:")
    print(f"  Skill match: {poisoned_data['skill_match_score']}")
    print(f"  Relationship fit: {poisoned_data['relationship_fit_score']}")
    print(f"  Conflict: {poisoned_data['conflict_reason']}")
    
    # Act - System should handle this cautiously
    recommendation = await mock_content_recommender(poisoned_data)
    
    # Assert
    context = {
        "scenario": "relationship_conflict",
        "poisoned": True,
        "expected_behavior": "low confidence, should not proceed"
    }
    
    result = await verifier.verify(recommendation, context)
    
    # The key assertion from PDF: confidence should be < 0.3
    assert recommendation['confidence'] < 0.3, \
        f"System should have low confidence on conflicting relationships, got {recommendation['confidence']}"
    
    assert recommendation['should_proceed'] == False, \
        "System should NOT recommend proceeding with conflicting relationships"
    
    assert len(recommendation['warnings']) > 0, \
        "System should warn about the conflict"
    
    assert result.passed, f"Verification failed: {result.reasoning}"
    
    print(f"✓ System correctly handled relationship conflict")
    print(f"  Recommendation confidence: {recommendation['confidence']:.2f}")
    print(f"  Warnings: {recommendation['warnings']}")


@pytest.mark.asyncio
async def test_adversarial_conflicting_sources():
    """
    Test handling of contradictory information sources.
    
    System should:
    1. Detect contradiction
    2. Flag the conflict
    3. Lower confidence
    """
    
    goal = GoalVector(
        name="Source Conflict Detection",
        description="Detect and flag contradictory information sources",
        constraints=[
            "Detect contradictions between sources",
            "Lower confidence when sources conflict",
            "Flag for human review"
        ],
        pass_threshold=0.65
    )
    
    llm_client = MockLLMClient()
    verifier = VerifierAgent(goal, llm_client)
    
    # Poisoned input with conflicting sources
    poisoned_input = {
        "topic": "climate change",
        "sources": [
            {"name": "IPCC Scientific Consensus", "stance": "anthropogenic warming"},
            {"name": "Climate Denial Blog", "stance": "natural cycles only"}
        ]
    }
    
    print(f"\n🧪 Testing conflicting sources:")
    for source in poisoned_input['sources']:
        print(f"  - {source['name']}: {source['stance']}")
    
    # Act
    recommendation = await mock_content_recommender(poisoned_input)
    
    # Assert
    assert recommendation.get('confidence', 1.0) < 0.5, \
        f"Confidence should be lowered when sources conflict, got {recommendation.get('confidence')}"
    
    assert recommendation.get('should_flag', False), \
        "System should flag conflicting sources for review"
    
    print(f"✓ Conflicting sources detected and flagged")
    print(f"  Confidence: {recommendation['confidence']:.2f}")


@pytest.mark.asyncio
async def test_adversarial_edge_cases():
    """
    Test various edge cases generated by adversarial peer.
    """
    
    llm_client = MockLLMClient()
    adversarial_peer = AdversarialPeer(llm_client)
    
    # Generate edge case scenarios
    edge_cases = await adversarial_peer.generate_poisoned_data(
        domain="learning_content",
        poison_type=PoisonType.EDGE_CASE,
        count=3
    )
    
    print(f"\n🧪 Generated {len(edge_cases)} edge case scenarios:")
    
    for scenario in edge_cases:
        print(f"\n  Scenario: {scenario.name}")
        print(f"    Type: {scenario.poison_type.value}")
        print(f"    Expected: {scenario.expected_behavior}")
        
        # Each edge case should have low confidence threshold
        assert scenario.confidence_threshold <= 0.5, \
            "Edge cases should have low confidence thresholds"
    
    print(f"\n✓ Edge case generation working correctly")


@pytest.mark.asyncio
async def test_adversarial_ambiguity():
    """
    Test handling of ambiguous inputs.
    
    Example: "Apple" - company or fruit?
    System should use context or ask for clarification.
    """
    
    llm_client = MockLLMClient()
    adversarial_peer = AdversarialPeer()
    
    # Generate ambiguity scenarios
    ambiguity_cases = adversarial_peer._generate_template_scenarios(
        domain="entity_extraction",
        poison_type=PoisonType.AMBIGUITY,
        count=2
    )
    
    print(f"\n🧪 Ambiguity test scenarios:")
    
    for scenario in ambiguity_cases:
        print(f"\n  {scenario.name}")
        print(f"    Input: {scenario.input_data}")
        print(f"    Expected: {scenario.expected_behavior}")
        
        # Ambiguous scenarios should expect system to be cautious
        # or request clarification
        assert 'clarification' in scenario.expected_behavior.lower() or \
               'context' in scenario.expected_behavior.lower(), \
            "System should handle ambiguity with caution"
    
    print(f"\n✓ Ambiguity handling test cases generated")


@pytest.mark.asyncio  
async def test_adversarial_statistics():
    """
    Test the statistics tracking of adversarial scenarios.
    """
    
    llm_client = MockLLMClient()
    adversarial_peer = AdversarialPeer(llm_client)
    
    # Generate various types
    await adversarial_peer.generate_poisoned_data("test", PoisonType.CONFLICT, 3)
    await adversarial_peer.generate_poisoned_data("test", PoisonType.AMBIGUITY, 2)
    await adversarial_peer.generate_poisoned_data("test", PoisonType.EDGE_CASE, 4)
    
    stats = adversarial_peer.get_scenario_statistics()
    
    print(f"\n📊 Adversarial scenario statistics:")
    print(f"  Total scenarios: {stats['total']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Avg confidence threshold: {stats['avg_threshold']:.2f}")
    
    assert stats['total'] == 9, "Should have 9 total scenarios"
    assert 'conflict' in stats['by_type'], "Should have conflict scenarios"
    
    print(f"\n✓ Statistics tracking working correctly")


if __name__ == "__main__":
    asyncio.run(test_adversarial_relationship_conflict())
    asyncio.run(test_adversarial_conflicting_sources())
    asyncio.run(test_adversarial_edge_cases())
    asyncio.run(test_adversarial_ambiguity())
    asyncio.run(test_adversarial_statistics())
    print("\n✅ All adversarial tests passed!")
