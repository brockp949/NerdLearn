"""
Example: TDFlow Test Generation

Demonstrates the Refiner Agent generating tests BEFORE content creation.

From PDF:
"Enforce a policy where the 'Refiner' agent generates the test suite
BEFORE the 'Instructional Design' agent generates the content."
"""

import pytest
import asyncio
from apps.testing.agents.refiner_agent import RefinerAgent, TDFlowPlan

class MockLLMClient:
    """Mock LLM for test generation"""
    async def generate(self, prompt: str, **kwargs) -> str:
        return '''```json
[
  {
    "name": "test_quadratic_roots",
    "description": "Verify calculation of roots for standard quadratic equation",
    "test_type": "unit",
    "target_component": "QuadraticSolver.solve",
    "input_specification": {"a": 1, "b": -3, "c": 2},
    "expected_behavior": "Should return [2.0, 1.0]",
    "coverage_score": 0.4,
    "triviality_score": 0.1
  },
  {
    "name": "test_quadratic_complex_roots",
    "description": "Verify handling of negative discriminant",
    "test_type": "boundary",
    "target_component": "QuadraticSolver.solve",
    "input_specification": {"a": 1, "b": 1, "c": 5},
    "expected_behavior": "Should raise ValueError or return complex numbers",
    "edge_cases": ["negative_discriminant"],
    "coverage_score": 0.3,
    "triviality_score": 0.0
  },
  {
    "name": "test_quadratic_linear_degenerate",
    "description": "Verify behavior when a=0 (linear equation)",
    "test_type": "boundary",
    "target_component": "QuadraticSolver.solve",
    "input_specification": {"a": 0, "b": 2, "c": -4},
    "expected_behavior": "Should solve linear equation 2x-4=0 -> x=2",
    "edge_cases": ["a_is_zero"],
    "coverage_score": 0.2,
    "triviality_score": 0.0
  }
]
```'''

@pytest.mark.asyncio
async def test_tdflow_generation():
    """Test standard TDFlow generation"""
    # Arrange
    llm_client = MockLLMClient()
    refiner = RefinerAgent(llm_client)
    
    component_spec = {
        "name": "QuadraticSolver",
        "requirements": "Solve ax^2+bx+c=0",
        "constraints": ["Handle complex roots", "Validate inputs"]
    }
    
    # Act
    plan = await refiner.generate_test_plan(component_spec)
    
    # Assert
    assert len(plan.tests) == 3, "Should generate 3 tests"
    assert plan.estimated_coverage >= 0.9, "Should have high estimated coverage"
    assert any(t.test_type.value == "boundary" for t in plan.tests), \
        "Should include boundary tests"
    
    print("\n📝 Generated TDFlow Plan:")
    for test in plan.tests:
        print(f"  - {test.name}: {test.description}")
        print(f"    Expected: {test.expected_behavior}")

@pytest.mark.asyncio
async def test_triviality_filtering():
    """Test that trivial tests are filtered out"""
    class TrivialLLMClient:
        async def generate(self, prompt: str, **kwargs) -> str:
            return '''```json
            [
              {
                "name": "test_exists",
                "description": "Check if class exists",
                "test_type": "unit",
                "triviality_score": 0.9,
                "coverage_score": 0.01
              },
              {
                "name": "test_real_logic",
                "description": "Test actual logic",
                "test_type": "unit",
                "triviality_score": 0.1,
                "coverage_score": 0.5
              }
            ]
            ```'''
            
    refiner = RefinerAgent(TrivialLLMClient())
    plan = await refiner.generate_test_plan({"name": "Test"})
    
    assert len(plan.tests) == 1, "Should filter out trivial test"
    assert plan.tests[0].name == "test_real_logic"
    print("\n✓ Triviality filtering working")

if __name__ == "__main__":
    asyncio.run(test_tdflow_generation())
    asyncio.run(test_triviality_filtering())
