"""
Adversarial Peer - Chaos Monkey for Agentic Testing

Generates adversarial test data to stress-test the system.
Implements the 'Chaos Monkey' concept - purposely inject 'poisoned' data
to test robustness.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging
import random

logger = logging.getLogger(__name__)


class PoisonType(Enum):
    """Types of adversarial scenarios"""
    CONFLICT = "conflict"  # Contradicting information
    AMBIGUITY = "ambiguity"  # Multiple valid interpretations
    SEMANTIC_TRAP = "semantic_trap"  # Correct syntax, wrong semantics
    EDGE_CASE = "edge_case"  # Boundary conditions
    CONTEXT_MISMATCH = "context_mismatch"  # Wrong context association


@dataclass
class AdversarialScenario:
    """Represents a 'poisoned' test scenario"""
    name: str
    description: str
    poison_type: PoisonType
    input_data: Dict[str, Any]
    expected_behavior: str
    confidence_threshold: float = 0.3  # System should be cautious
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdversarialPeer:
    """
    Generates adversarial test data to stress-test the system.
    
    From PDF:
    "Using the concepts from NerdLearn's 'Adversarial Peer,' we purposely
    inject 'poisoned' data to test the system's robustness."
    
    Example:
        The Adversarial Peer introduces two contacts with high influence
        but negative 'Relationship Fit' (e.g., they are competitors or
        have a history of conflict). The system should realize that
        introducing two enemies will kill the project, even if their
        skills are perfect.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize Adversarial Peer.
        
        Args:
            llm_client: Optional LLM for generating creative adversarial scenarios
        """
        self.llm = llm_client
        self.scenarios: List[AdversarialScenario] = []
    
    async def generate_poisoned_data(
        self,
        domain: str,
        poison_type: PoisonType,
        count: int = 10,
        constraints: Optional[List[str]] = None
    ) -> List[AdversarialScenario]:
        """
        Generate adversarial test data using LLM creativity.
        
        Args:
            domain: Domain context (e.g., 'content_recommendation', 'learning_path')
            poison_type: Type of adversarial scenario
            count: Number of scenarios to generate
            constraints: Optional constraints for generation
        
        Returns:
            List of adversarial scenarios
        
        Examples of generated scenarios:
        - Ambiguous entities (Apple company vs apple fruit)
        - Conflicting relationships (competitors suggested as partners)
        - Edge cases (empty strings, extreme values)
        - Semantic traps (correct syntax, wrong semantics)
        """
        if self.llm is None:
            logger.warning("No LLM client provided, using template-based generation")
            return self._generate_template_scenarios(domain, poison_type, count)
        
        logger.info(f"Generating {count} {poison_type.value} scenarios for {domain}")
        
        constraint_str = ', '.join(constraints) if constraints else 'None'
        
        prompt = f"""Generate {count} adversarial test cases for domain: {domain}

POISON TYPE: {poison_type.value}
CONSTRAINTS: {constraint_str}

Each test case should:
1. Be syntactically valid
2. Contain a semantic trap or edge case
3. Test the system's robustness and error handling
4. Include the expected failure mode or cautious behavior

Respond in JSON array format:
[
    {{
        "name": "descriptive scenario name",
        "description": "what makes this adversarial",
        "input": {{"field": "value"}},
        "expected_behavior": "how the system should handle this",
        "confidence_threshold": 0.0-1.0
    }}
]

Make these scenarios creative and challenging. Think like a penetration tester.
"""
        
        try:
            response = await self.llm.generate(prompt)
            scenarios_data = self._parse_llm_response(response)
            
            scenarios = []
            for data in scenarios_data:
                scenario = AdversarialScenario(
                    name=data.get('name', f'scenario_{len(scenarios)}'),
                    description=data.get('description', ''),
                    poison_type=poison_type,
                    input_data=data.get('input', {}),
                    expected_behavior=data.get('expected_behavior', ''),
                    confidence_threshold=data.get('confidence_threshold', 0.3)
                )
                scenarios.append(scenario)
                self.scenarios.append(scenario)
            
            logger.info(f"Generated {len(scenarios)} adversarial scenarios")
            return scenarios
            
        except Exception as e:
            logger.error(f"Failed to generate scenarios: {e}")
            return self._generate_template_scenarios(domain, poison_type, count)
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract scenarios"""
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract from markdown code block
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            elif '```' in response:
                json_start = response.find('```') + 3
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            raise ValueError("Could not parse LLM response")
    
    def _generate_template_scenarios(
        self,
        domain: str,
        poison_type: PoisonType,
        count: int
    ) -> List[AdversarialScenario]:
        """
        Generate scenarios using templates when LLM is unavailable.
        
        This provides basic adversarial testing without requiring LLM calls.
        """
        templates = {
            PoisonType.AMBIGUITY: [
                {
                    'name': 'Ambiguous entity reference',
                    'description': 'Entity name has multiple meanings',
                    'input': {'text': 'Apple released new products', 'context': 'general'},
                    'expected_behavior': 'Should ask for clarification or use context clues'
                },
                {
                    'name': 'Homonym challenge',
                    'description': 'Word with multiple meanings',
                    'input': {'text': 'Bank on the river', 'context': 'finance'},
                    'expected_behavior': 'Should recognize context mismatch'
                }
            ],
            PoisonType.CONFLICT: [
                {
                    'name': 'Conflicting requirements',
                    'description': 'Two contradicting constraints',
                    'input': {'requirement_a': 'maximize speed', 'requirement_b': 'maximize accuracy'},
                    'expected_behavior': 'Should identify trade-off and request prioritization'
                },
                {
                    'name': 'Contradicting sources',
                    'description': 'Two sources with opposite information',
                    'input': {'source_1': 'X is true', 'source_2': 'X is false'},
                    'expected_behavior': 'Should flag conflict and lower confidence'
                }
            ],
            PoisonType.EDGE_CASE: [
                {
                    'name': 'Empty input',
                    'description': 'Input fields are empty',
                    'input': {'text': '', 'context': ''},
                    'expected_behavior': 'Should handle gracefully with error message'
                },
                {
                    'name': 'Extreme values',
                    'description': 'Values outside normal range',
                    'input': {'score': 999999, 'difficulty': -100},
                    'expected_behavior': 'Should validate and reject or normalize'
                }
            ],
            PoisonType.SEMANTIC_TRAP: [
                {
                    'name': 'Valid syntax, invalid semantics',
                    'description': 'Grammatically correct but nonsensical',
                    'input': {'text': 'The colorless green ideas sleep furiously'},
                    'expected_behavior': 'Should recognize semantic invalidity'
                }
            ],
            PoisonType.CONTEXT_MISMATCH: [
                {
                    'name': 'Wrong context association',
                    'description': 'Content doesn\'t match declared context',
                    'input': {'text': 'Quantum mechanics equations', 'context': 'cooking'},
                    'expected_behavior': 'Should detect context mismatch'
                }
            ]
        }
        
        template_list = templates.get(poison_type, templates[PoisonType.EDGE_CASE])
        
        scenarios = []
        for i in range(min(count, len(template_list))):
            template = template_list[i % len(template_list)]
            scenario = AdversarialScenario(
                name=f"{template['name']}_{i}",
                description=template['description'],
                poison_type=poison_type,
                input_data=template['input'],
                expected_behavior=template['expected_behavior'],
                confidence_threshold=0.3
            )
            scenarios.append(scenario)
            self.scenarios.append(scenario)
        
        return scenarios
    
    def inject_relationship_conflict(
        self,
        data: Dict[str, Any],
        conflict_reason: str = 'competitors'
    ) -> Dict[str, Any]:
        """
        Inject conflicting relationship data.
        
        From PDF example:
        "Two contacts with high skill match but negative relationship fit
        (e.g., they are competitors or have a history of conflict).
        System should realize that introducing two enemies will kill the
        project, even if their skills are perfect."
        
        Args:
            data: Original data to poison
            conflict_reason: Reason for conflict
        
        Returns:
            Poisoned data with conflict markers
        """
        poisoned = data.copy()
        
        # Add conflicting relationship indicators
        poisoned['relationship_fit_score'] = round(random.uniform(-0.9, -0.5), 2)
        poisoned['skill_match_score'] = round(random.uniform(0.85, 0.98), 2)
        poisoned['conflict_reason'] = conflict_reason
        poisoned['collaboration_history'] = 'negative'
        
        # Mark as adversarial
        poisoned['_adversarial'] = {
            'type': 'relationship_conflict',
            'expected_confidence': '<0.3',
            'reason': f'High skill match but {conflict_reason}'
        }
        
        return poisoned
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """Get statistics on generated scenarios"""
        if not self.scenarios:
            return {'total': 0, 'by_type': {}}
        
        by_type = {}
        for scenario in self.scenarios:
            poison_type = scenario.poison_type.value
            by_type[poison_type] = by_type.get(poison_type, 0) + 1
        
        return {
            'total': len(self.scenarios),
            'by_type': by_type,
            'avg_threshold': sum(s.confidence_threshold for s in self.scenarios) / len(self.scenarios)
        }
