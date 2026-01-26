"""
Base Verifier Agent for semantic test validation.

Implements the 'Antigravity' concept - uses goal vectors to focus testing
on semantic correctness beyond syntactic validity.
"""

from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GoalVector:
    """
    Represents a testing objective with semantic embedding.
    
    The goal vector provides 'gravitational bias' - it keeps the test
    focused on what matters, preventing 'ADHD' behavior in AI validation.
    """
    name: str
    description: str
    embedding: Optional[np.ndarray] = None
    constraints: List[str] = field(default_factory=list)
    pass_threshold: float = 0.7
    
    def __post_init__(self):
        """Initialize embedding if not provided"""
        if self.embedding is None:
            # Placeholder - will be replaced with actual embedding generation
            self.embedding = np.zeros(768)


class LLMClient(Protocol):
    """Protocol for LLM client interface"""
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        ...


@dataclass
class VerificationResult:
    """Result of a verification check"""
    passed: bool
    confidence: float
    reasoning: str
    violations: List[str] = field(default_factory=list)
    goal_alignment: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'passed': self.passed,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'violations': self.violations,
            'goal_alignment': self.goal_alignment,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class VerifierAgent:
    """
    Base class for semantic test verification using LLM-powered auditing.
    
    This implements the core 'Antigravity' testing concept:
    - Standard tests check syntax
    - Verifier tests check semantic correctness with goal alignment
    
    Example from PDF:
        Standard test: Check if 'Apple' is extracted as Organization
        Antigravity test: Check if 'Apple' is extracted AND linked to
                         'Consumer Electronics' context, filtering out
                         'apple' (the fruit)
    """
    
    def __init__(self, goal_vector: GoalVector, llm_client: LLMClient):
        """
        Initialize verifier agent.
        
        Args:
            goal_vector: The testing objective with constraints
            llm_client: LLM interface for semantic validation
        """
        self.goal = goal_vector
        self.llm = llm_client
        self.audit_history: List[VerificationResult] = []
    
    async def verify(
        self,
        test_output: Any,
        context: Optional[Dict[str, Any]] = None,
        expected: Optional[Any] = None
    ) -> VerificationResult:
        """
        Perform semantic verification of test output.
        
        This is the core method that distinguishes agentic testing from
        traditional testing. It uses an LLM 'auditor' to check semantic
        correctness, not just syntactic validity.
        
        Args:
            test_output: The output to verify
            context: Additional context for verification
            expected: Optional expected output for comparison
        
        Returns:
            VerificationResult with pass/fail, confidence, reasoning
        """
        logger.info(f"Verifying output against goal: {self.goal.name}")
        
        try:
            # Build verification prompt with gravitational biasing
            prompt = self._build_verification_prompt(test_output, context, expected)
            
            # Get LLM audit
            audit_response = await self.llm.generate(prompt)
            
            # Parse and validate
            result = self._parse_audit_response(audit_response)
            
            # Apply gravitational bias - check goal alignment
            goal_alignment = await self._compute_goal_alignment(test_output, context)
            result.goal_alignment = goal_alignment
            
            # Final pass decision combines LLM verdict and goal alignment
            result.passed = result.passed and (goal_alignment >= self.goal.pass_threshold)
            
            if not result.passed:
                logger.warning(
                    f"Verification failed: {result.reasoning} "
                    f"(alignment: {goal_alignment:.2f})"
                )
            
            # Store in audit history
            self.audit_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed with error: {e}")
            return VerificationResult(
                passed=False,
                confidence=0.0,
                reasoning=f"Verification error: {str(e)}",
                violations=['system_error']
            )
    
    def _build_verification_prompt(
        self,
        output: Any,
        context: Optional[Dict],
        expected: Optional[Any]
    ) -> str:
        """
        Construct the auditor prompt with goal biasing.
        
        This prompt structure implements 'Gravitational Biasing' by:
        1. Clearly stating the goal vector
        2. Listing explicit constraints
        3. Requesting semantic correctness evaluation
        4. Demanding structured output with confidence scores
        """
        context_str = json.dumps(context, indent=2) if context else 'None provided'
        expected_str = json.dumps(expected, indent=2) if expected else 'Not specified'
        
        # Handle different output types
        if isinstance(output, (dict, list)):
            output_str = json.dumps(output, indent=2)
        else:
            output_str = str(output)
        
        return f"""You are a Test Auditor Agent. Your role is to verify test output quality using semantic analysis.

GOAL VECTOR: {self.goal.name}
DESCRIPTION: {self.goal.description}
CONSTRAINTS: {', '.join(self.goal.constraints) if self.goal.constraints else 'None'}

TEST OUTPUT:
{output_str}

EXPECTED OUTPUT:
{expected_str}

CONTEXT:
{context_str}

VERIFICATION TASK:
Analyze the test output and determine if it meets the goal vector requirements.

Consider:
1. **Goal Alignment**: Does the output align with the stated goal?
2. **Constraint Compliance**: Are all constraints satisfied?
3. **Semantic Correctness**: Is the output semantically correct beyond just syntactic validity?
4. **Context Awareness**: Does the output demonstrate proper understanding of context?

Rate your confidence in the output quality from 0.0 (completely incorrect) to 1.0 (perfect).

Respond ONLY in valid JSON format:
{{
    "passed": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "detailed explanation of your assessment",
    "violations": ["list any constraint violations, empty array if none"]
}}"""
    
    async def _compute_goal_alignment(
        self,
        output: Any,
        context: Optional[Dict]
    ) -> float:
        """
        Compute embedding similarity between output and goal vector.
        
        This is the 'Antigravity' gravitational bias component.
        It measures how well the output aligns with the intended goal
        using semantic embeddings.
        
        TODO: Implement actual embedding generation using NerdLearn's
        existing embedding service once integrated.
        
        Returns:
            Float from 0.0 to 1.0 representing alignment
        """
        # Placeholder implementation
        # In production, this would:
        # 1. Generate embedding for output
        # 2. Compute cosine similarity with goal.embedding
        # 3. Return normalized similarity score
        
        # For now, return a reasonable default
        # This will be replaced with actual implementation
        return 0.80  # Default moderate alignment
    
    def _parse_audit_response(self, response: str) -> VerificationResult:
        """
        Parse LLM audit response into structured result.
        
        Handles various response formats and extraction of the JSON block.
        """
        try:
            # Try to parse as direct JSON
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            try:
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    json_str = response[json_start:json_end].strip()
                    data = json.loads(json_str)
                elif '```' in response:
                    json_start = response.find('```') + 3
                    json_end = response.find('```', json_start)
                    json_str = response[json_start:json_end].strip()
                    data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse audit response: {e}")
                return VerificationResult(
                    passed=False,
                    confidence=0.0,
                    reasoning=f'Failed to parse audit response: {str(e)}',
                    violations=['parse_error']
                )
        
        # Extract fields with defaults
        return VerificationResult(
            passed=data.get('passed', False),
            confidence=float(data.get('confidence', 0.0)),
            reasoning=data.get('reasoning', 'No reasoning provided'),
            violations=data.get('violations', [])
        )
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from audit history.
        
        Useful for understanding test patterns and failure modes.
        """
        if not self.audit_history:
            return {
                'total_audits': 0,
                'pass_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_goal_alignment': 0.0
            }
        
        total = len(self.audit_history)
        passed = sum(1 for r in self.audit_history if r.passed)
        
        return {
            'total_audits': total,
            'pass_rate': passed / total,
            'avg_confidence': np.mean([r.confidence for r in self.audit_history]),
            'avg_goal_alignment': np.mean([r.goal_alignment for r in self.audit_history]),
            'common_violations': self._get_common_violations()
        }
    
    def _get_common_violations(self) -> Dict[str, int]:
        """Get frequency count of violations"""
        violations = {}
        for result in self.audit_history:
            for violation in result.violations:
                violations[violation] = violations.get(violation, 0) + 1
        return dict(sorted(violations.items(), key=lambda x: x[1], reverse=True))
