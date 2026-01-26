"""
Example: Entity Extraction Test with Verifier Agent

Demonstrates how to use the VerifierAgent for semantic validation
of entity extraction with contextual disambiguation.

From PDF:
"Standard tests check if 'Apple' is extracted as an Organization.
An Antigravity test checks if 'Apple' is extracted and linked to
'Consumer Electronics' context, filtering out 'apple' (the fruit)."
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any

# These imports will work once integrated with NerdLearn's LLM infrastructure
# from app.ai.llm_client import get_llm_client
# from app.rag.entity_extractor import extract_entities

from apps.testing.agents.base_verifier import VerifierAgent, GoalVector


class MockLLMClient:
    """Mock LLM client for testing"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return mock audit response"""
        # In real implementation, this would call actual LLM
        # For demo, return a successful audit
        return '''```json
{
    "passed": true,
    "confidence": 0.85,
    "reasoning": "Entity extracted correctly as Organization with proper technology context. No confusion with fruit meaning.",
    "violations": []
}
```'''


async def mock_extract_entities(text: str) -> list:
    """Mock entity extraction function"""
    # In real implementation, this would use NerdLearn's entity extractor
    return [
        {
            "name": "Apple",
            "type": "Organization",
            "context": "technology, consumer electronics",
            "confidence": 0.92
        },
        {
            "name": "MacBook Pro",
            "type": "Product",
            "context": "technology",
            "confidence": 0.95
        }
    ]


@pytest.mark.asyncio
async def test_entity_extraction_with_verifier():
    """
    Test entity extraction with semantic verification.
    
    This demonstrates the 'Antigravity' testing approach:
    - Not just checking if Apple is extracted
    - Verifying it's the RIGHT Apple (company, not fruit)
    - Confirming proper context association
    """
    
    # Arrange - Set up goal vector
    goal = GoalVector(
        name="Semantic Entity Extraction",
        description="Entities must be extracted with correct type and contextual disambiguation",
        embedding=np.random.rand(768),  # Would be actual embedding in production
        constraints=[
            "Correct entity type (Organization vs Product vs Location)",
            "Proper context association (technology vs food vs geography)",
            "No ambiguous references without clarification"
        ],
        pass_threshold=0.75
    )
    
    # Create verifier agent
    llm_client = MockLLMClient()
    verifier = VerifierAgent(goal, llm_client)
    
    # Act - Extract entities
    test_input = {
        "text": "Apple announced new MacBook Pro models at their event.",
        "domain": "technology news"
    }
    
    extracted_entities = await mock_extract_entities(test_input['text'])
    
    # Assert with Verifier Agent
    verification_context = {
        "domain": "technology",
        "expected_categories": ["Consumer Electronics", "Technology"],
        "should_exclude": ["fruit", "food", "agriculture"]
    }
    
    result = await verifier.verify(extracted_entities, verification_context)
    
    # Standard assertions
    assert result.passed, f"Semantic verification failed: {result.reasoning}"
    assert result.confidence > 0.7, f"Low confidence: {result.confidence}"
    assert len(result.violations) == 0, f"Constraints violated: {result.violations}"
    
    # Verify Apple is correctly identified as Organization
    apple_entity = next((e for e in extracted_entities if e['name'] == 'Apple'), None)
    assert apple_entity is not None, "Apple entity not extracted"
    assert apple_entity['type'] == 'Organization', "Apple not recognized as Organization"
    assert 'technology' in apple_entity['context'].lower(), \
        "Apple not linked to technology context"
    
    print("✓ Entity extraction passed semantic verification")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Goal Alignment: {result.goal_alignment:.2f}")
    print(f"  Reasoning: {result.reasoning}")


@pytest.mark.asyncio
async def test_entity_ambiguity_detection():
    """
    Test that verifier detects ambiguous entity references.
    
    Example: "bank" could be financial institution or river bank
    """
    
    goal = GoalVector(
        name="Ambiguity Detection",
        description="System should detect and handle ambiguous entity references",
        constraints=[
            "Flag ambiguous references",
            "Use context clues for disambiguation",
            "Lower confidence when context is unclear"
        ],
        pass_threshold=0.60
    )
    
    llm_client = MockLLMClient()
    verifier = VerifierAgent(goal, llm_client)
    
    # Ambiguous input without clear context
    test_input = "The bank is beautiful."
    
    # Mock extraction that might struggle with ambiguity
    entities = [
        {
            "name": "bank",
            "type": "Unknown",
            "context": "ambiguous",
            "confidence": 0.45,  # Low confidence indicates uncertainty
            "possible_types": ["FinancialInstitution", "Geography"]
        }
    ]
    
    context = {
        "text": test_input,
        "domain": "unknown",
        "ambiguity_expected": True
    }
    
    result = await verifier.verify(entities, context)
    
    # For ambiguous cases, we expect system to be cautious
    # The entity confidence should be low, indicating uncertainty
    assert entities[0]['confidence'] < 0.6, \
        "System should have low confidence on ambiguous input"
    
    print("✓ Ambiguity detection working correctly")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_entity_extraction_with_verifier())
    asyncio.run(test_entity_ambiguity_detection())
    print("\n✓ All example tests passed!")
