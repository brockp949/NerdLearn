"""
Agentic Code Evaluation System

Research alignment:
- Immediate Feedback: Reduces time between submission and learning
- Formative Assessment: Focus on improvement, not just grading
- Multi-dimensional Evaluation: Beyond just correctness
- Scaffolded Hints: Progressive support without giving away answers
- Code Review as Learning: Professional practices build transferable skills

Multiple AI agents evaluate code submissions from different perspectives:
1. Correctness Agent: Does it work?
2. Quality Agent: Is it clean and maintainable?
3. Efficiency Agent: Is it performant?
4. Security Agent: Is it safe?
5. Teaching Agent: What can they learn?
"""
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import logging
import json
import asyncio
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)


class DifficultyLevel(str, Enum):
    """Challenge difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class EvaluationDimension(str, Enum):
    """Dimensions of code evaluation"""
    CORRECTNESS = "correctness"       # Does it produce correct output?
    QUALITY = "quality"               # Is it clean, readable, idiomatic?
    EFFICIENCY = "efficiency"         # Time/space complexity
    SECURITY = "security"             # Potential vulnerabilities
    COMPLETENESS = "completeness"     # Edge cases handled?
    DOCUMENTATION = "documentation"   # Comments and docstrings


class FeedbackType(str, Enum):
    """Types of feedback to provide"""
    PRAISE = "praise"           # What they did well
    ISSUE = "issue"             # Problems found
    SUGGESTION = "suggestion"   # How to improve
    HINT = "hint"               # Guided toward solution
    EXAMPLE = "example"         # Code example


class HintLevel(str, Enum):
    """Progressive hint levels (scaffolding)"""
    NUDGE = "nudge"           # Very subtle direction
    GUIDANCE = "guidance"      # More specific direction
    EXPLANATION = "explanation"  # Detailed explanation
    PARTIAL = "partial"        # Partial solution
    SOLUTION = "solution"      # Full solution


@dataclass
class TestCase:
    """Test case for a coding challenge"""
    input_data: Any
    expected_output: Any
    description: str
    is_hidden: bool = False  # Hidden test cases for final evaluation
    weight: float = 1.0
    edge_case: bool = False
    timeout_seconds: float = 5.0


@dataclass
class CodingChallenge:
    """Complete coding challenge definition"""
    challenge_id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    language: str  # python, javascript, etc.

    # Problem specification
    function_name: str
    parameters: List[Dict[str, str]]  # [{"name": "n", "type": "int"}]
    return_type: str
    constraints: List[str]

    # Test cases
    test_cases: List[TestCase]

    # Learning objectives
    concepts_tested: List[str]
    skills_required: List[str]

    # Hints (progressive scaffolding)
    hints: List[str]

    # Solution for reference (not shown to learner)
    reference_solution: str
    time_complexity: str
    space_complexity: str

    # Metadata
    estimated_minutes: int = 15
    tags: List[str] = field(default_factory=list)


@dataclass
class FeedbackItem:
    """Single piece of feedback"""
    type: FeedbackType
    dimension: EvaluationDimension
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension"""
    dimension: EvaluationDimension
    score: float  # 0-100
    feedback: List[FeedbackItem]
    strengths: List[str]
    improvements: List[str]


@dataclass
class EvaluationResult:
    """Complete evaluation of a submission"""
    submission_id: str
    challenge_id: str
    user_id: str

    # Overall results
    passed: bool
    overall_score: float  # 0-100
    tests_passed: int
    tests_total: int

    # Dimension scores
    dimension_scores: Dict[EvaluationDimension, DimensionScore]

    # Detailed feedback
    feedback_items: List[FeedbackItem]

    # Learning insights
    concepts_demonstrated: List[str]
    concepts_to_review: List[str]
    next_challenge_suggestion: Optional[str]

    # Execution details
    execution_time_ms: Optional[float]
    memory_used_kb: Optional[float]
    runtime_errors: List[str]

    # Metadata
    evaluation_time: datetime = field(default_factory=datetime.utcnow)


class AgentEvaluation(BaseModel):
    """Structured evaluation from an AI agent"""
    score: float = Field(ge=0, le=100, description="Score 0-100")
    passed: bool = Field(description="Whether this dimension passes")
    strengths: List[str] = Field(default_factory=list)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Agent's reasoning")


class CodeEvaluator:
    """
    Multi-agent code evaluation system

    Evaluates code submissions from multiple perspectives using
    specialized AI agents for different quality dimensions.
    """

    # Agent prompts for each dimension
    AGENT_PROMPTS = {
        EvaluationDimension.CORRECTNESS: """You are a Correctness Evaluator for code submissions.

Your role is to assess whether the code:
1. Produces correct output for all test cases
2. Handles edge cases properly
3. Meets the problem specification exactly
4. Returns the correct data types

Score 0-100 where:
- 100: All tests pass, including edge cases
- 80-99: Most tests pass, minor issues
- 60-79: Core functionality works but significant issues
- 40-59: Partially correct
- 0-39: Fundamentally broken

Be specific about what's wrong and how to fix it.""",

        EvaluationDimension.QUALITY: """You are a Code Quality Evaluator.

Your role is to assess:
1. Code readability and clarity
2. Proper naming conventions
3. Code organization and structure
4. Idiomatic usage of the language
5. Appropriate abstraction level
6. DRY (Don't Repeat Yourself) principle

Score 0-100 where:
- 100: Exemplary, production-ready code
- 80-99: Clean code with minor style issues
- 60-79: Readable but could be cleaner
- 40-59: Messy but functional
- 0-39: Difficult to read and maintain

Focus on teaching good practices, not nitpicking.""",

        EvaluationDimension.EFFICIENCY: """You are an Efficiency Evaluator.

Your role is to assess:
1. Time complexity (Big O)
2. Space complexity
3. Unnecessary operations
4. Algorithm choice
5. Data structure usage

Score 0-100 where:
- 100: Optimal solution
- 80-99: Near-optimal, minor inefficiencies
- 60-79: Acceptable but room for improvement
- 40-59: Noticeably inefficient
- 0-39: Very inefficient approach

Explain complexity in accessible terms for learners.""",

        EvaluationDimension.SECURITY: """You are a Security Evaluator.

Your role is to identify:
1. Input validation issues
2. Potential injection vulnerabilities
3. Resource exhaustion risks
4. Error handling gaps
5. Unsafe operations

Score 0-100 where:
- 100: No security concerns
- 80-99: Minor considerations
- 60-79: Some vulnerabilities to address
- 40-59: Significant security issues
- 0-39: Critical vulnerabilities

Note: For learning exercises, focus on teaching good habits.""",

        EvaluationDimension.DOCUMENTATION: """You are a Documentation Evaluator.

Your role is to assess:
1. Function/method docstrings
2. Inline comments where needed
3. Variable naming clarity
4. Code self-documentation

Score 0-100 where:
- 100: Perfectly documented
- 80-99: Well documented
- 60-79: Adequate documentation
- 40-59: Minimal documentation
- 0-39: No meaningful documentation

Focus on teaching when and how to document."""
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,  # Lower for consistent evaluation
        llm: Optional[ChatOpenAI] = None
    ):
        """Initialize code evaluator"""
        self.llm = llm or ChatOpenAI(model=model, temperature=temperature)
        self.challenges: Dict[str, CodingChallenge] = {}

    def register_challenge(self, challenge: CodingChallenge) -> None:
        """Register a coding challenge"""
        self.challenges[challenge.challenge_id] = challenge
        logger.info(f"Registered challenge: {challenge.title}")

    def create_challenge(
        self,
        challenge_id: str,
        title: str,
        description: str,
        difficulty: DifficultyLevel,
        function_name: str,
        parameters: List[Dict[str, str]],
        return_type: str,
        test_cases: List[Dict[str, Any]],
        concepts_tested: List[str],
        hints: List[str],
        reference_solution: str,
        language: str = "python"
    ) -> CodingChallenge:
        """
        Create and register a new coding challenge

        Args:
            challenge_id: Unique challenge identifier
            title: Challenge title
            description: Problem description
            difficulty: Difficulty level
            function_name: Expected function name
            parameters: List of parameter definitions
            return_type: Expected return type
            test_cases: List of test case dicts
            concepts_tested: Concepts this challenge tests
            hints: Progressive hints
            reference_solution: Reference implementation
            language: Programming language

        Returns:
            Created challenge
        """
        # Convert test case dicts to TestCase objects
        tc_objects = [
            TestCase(
                input_data=tc["input"],
                expected_output=tc["expected"],
                description=tc.get("description", ""),
                is_hidden=tc.get("hidden", False),
                edge_case=tc.get("edge_case", False)
            )
            for tc in test_cases
        ]

        challenge = CodingChallenge(
            challenge_id=challenge_id,
            title=title,
            description=description,
            difficulty=difficulty,
            language=language,
            function_name=function_name,
            parameters=parameters,
            return_type=return_type,
            constraints=[],
            test_cases=tc_objects,
            concepts_tested=concepts_tested,
            skills_required=[],
            hints=hints,
            reference_solution=reference_solution,
            time_complexity="O(?)",
            space_complexity="O(?)"
        )

        self.register_challenge(challenge)
        return challenge

    async def evaluate_submission(
        self,
        challenge_id: str,
        user_id: str,
        code: str,
        dimensions: Optional[List[EvaluationDimension]] = None
    ) -> EvaluationResult:
        """
        Evaluate a code submission

        Args:
            challenge_id: Challenge being attempted
            user_id: Submitting user
            code: Submitted code
            dimensions: Which dimensions to evaluate (default: all)

        Returns:
            Complete evaluation result
        """
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found")

        dimensions = dimensions or list(EvaluationDimension)
        submission_id = f"{user_id}_{challenge_id}_{datetime.utcnow().timestamp()}"

        # Run test cases first
        test_results, runtime_errors, execution_time = await self._run_tests(
            code, challenge
        )

        tests_passed = sum(1 for r in test_results if r["passed"])
        tests_total = len(test_results)

        # Run AI evaluations in parallel
        evaluation_tasks = [
            self._evaluate_dimension(dimension, code, challenge, test_results)
            for dimension in dimensions
        ]
        dimension_evaluations = await asyncio.gather(*evaluation_tasks)

        # Combine results
        dimension_scores = {}
        all_feedback = []

        for dimension, evaluation in zip(dimensions, dimension_evaluations):
            score = DimensionScore(
                dimension=dimension,
                score=evaluation.score,
                feedback=[],
                strengths=evaluation.strengths,
                improvements=evaluation.suggestions
            )
            dimension_scores[dimension] = score

            # Convert issues to feedback items
            for issue in evaluation.issues:
                feedback = FeedbackItem(
                    type=FeedbackType.ISSUE,
                    dimension=dimension,
                    message=issue.get("message", ""),
                    line_number=issue.get("line"),
                    suggestion=issue.get("fix")
                )
                all_feedback.append(feedback)
                score.feedback.append(feedback)

        # Calculate overall score (weighted average)
        weights = {
            EvaluationDimension.CORRECTNESS: 0.40,
            EvaluationDimension.QUALITY: 0.25,
            EvaluationDimension.EFFICIENCY: 0.20,
            EvaluationDimension.SECURITY: 0.10,
            EvaluationDimension.DOCUMENTATION: 0.05
        }

        overall_score = sum(
            dimension_scores[d].score * weights.get(d, 0.1)
            for d in dimension_scores
        )

        # Determine pass/fail (minimum 60% and all tests passing)
        passed = overall_score >= 60 and tests_passed == tests_total

        # Identify concepts demonstrated and to review
        concepts_demonstrated = []
        concepts_to_review = []

        for concept in challenge.concepts_tested:
            if overall_score >= 70:
                concepts_demonstrated.append(concept)
            else:
                concepts_to_review.append(concept)

        result = EvaluationResult(
            submission_id=submission_id,
            challenge_id=challenge_id,
            user_id=user_id,
            passed=passed,
            overall_score=overall_score,
            tests_passed=tests_passed,
            tests_total=tests_total,
            dimension_scores=dimension_scores,
            feedback_items=all_feedback,
            concepts_demonstrated=concepts_demonstrated,
            concepts_to_review=concepts_to_review,
            next_challenge_suggestion=None,  # Could be enhanced
            execution_time_ms=execution_time,
            memory_used_kb=None,
            runtime_errors=runtime_errors
        )

        logger.info(
            f"Evaluated {submission_id}: {overall_score:.1f}% "
            f"({tests_passed}/{tests_total} tests)"
        )

        return result

    async def _run_tests(
        self,
        code: str,
        challenge: CodingChallenge
    ) -> Tuple[List[Dict[str, Any]], List[str], Optional[float]]:
        """
        Run test cases against submitted code

        Returns:
            Tuple of (test_results, runtime_errors, execution_time_ms)
        """
        test_results = []
        runtime_errors = []
        total_time = 0.0

        for i, test in enumerate(challenge.test_cases):
            result = {
                "test_number": i + 1,
                "description": test.description,
                "passed": False,
                "expected": test.expected_output,
                "actual": None,
                "error": None
            }

            try:
                # Create a test script
                test_code = f"""
{code}

# Test execution
import json
import time
start = time.time()
result = {challenge.function_name}({self._format_input(test.input_data)})
elapsed = (time.time() - start) * 1000
print(json.dumps({{"result": result, "time_ms": elapsed}}))
"""
                # Execute in subprocess with timeout
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False
                ) as f:
                    f.write(test_code)
                    f.flush()

                    try:
                        proc = subprocess.run(
                            ['python', f.name],
                            capture_output=True,
                            text=True,
                            timeout=test.timeout_seconds
                        )

                        if proc.returncode == 0:
                            output = json.loads(proc.stdout.strip())
                            result["actual"] = output["result"]
                            total_time += output["time_ms"]

                            # Check correctness
                            if result["actual"] == test.expected_output:
                                result["passed"] = True
                        else:
                            result["error"] = proc.stderr.strip()
                            if result["error"]:
                                runtime_errors.append(result["error"])

                    except subprocess.TimeoutExpired:
                        result["error"] = f"Timeout ({test.timeout_seconds}s)"
                        runtime_errors.append(f"Test {i+1}: Timeout")

                    finally:
                        os.unlink(f.name)

            except Exception as e:
                result["error"] = str(e)
                runtime_errors.append(str(e))

            test_results.append(result)

        return test_results, runtime_errors, total_time if total_time > 0 else None

    def _format_input(self, input_data: Any) -> str:
        """Format input data for test script"""
        if isinstance(input_data, str):
            return f'"{input_data}"'
        elif isinstance(input_data, (list, dict)):
            return json.dumps(input_data)
        elif isinstance(input_data, tuple):
            # Unpack tuple as arguments
            formatted = [self._format_input(x) for x in input_data]
            return ", ".join(formatted)
        else:
            return str(input_data)

    async def _evaluate_dimension(
        self,
        dimension: EvaluationDimension,
        code: str,
        challenge: CodingChallenge,
        test_results: List[Dict[str, Any]]
    ) -> AgentEvaluation:
        """Get evaluation for a specific dimension from AI agent"""

        system_prompt = self.AGENT_PROMPTS.get(dimension, "Evaluate this code.")

        # Build context
        context = f"""
CHALLENGE: {challenge.title}
DIFFICULTY: {challenge.difficulty.value}
LANGUAGE: {challenge.language}

PROBLEM DESCRIPTION:
{challenge.description}

EXPECTED FUNCTION: {challenge.function_name}
PARAMETERS: {json.dumps(challenge.parameters)}
RETURN TYPE: {challenge.return_type}

CONCEPTS BEING TESTED: {', '.join(challenge.concepts_tested)}

SUBMITTED CODE:
```{challenge.language}
{code}
```

TEST RESULTS:
{json.dumps(test_results, indent=2)}

REFERENCE SOLUTION (for comparison, not to share with learner):
```{challenge.language}
{challenge.reference_solution}
```

Evaluate this submission for {dimension.value}.

Respond with JSON:
{{
    "score": <0-100>,
    "passed": <true/false>,
    "strengths": ["what they did well"],
    "issues": [
        {{"message": "description", "line": <line_number or null>, "fix": "suggestion"}}
    ],
    "suggestions": ["improvement suggestions"],
    "reasoning": "your evaluation reasoning"
}}
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ])

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            return AgentEvaluation(**result)
        except Exception as e:
            logger.warning(f"Failed to parse {dimension} evaluation: {e}")
            return AgentEvaluation(
                score=50,
                passed=False,
                reasoning="Evaluation parsing failed"
            )

    async def get_hint(
        self,
        challenge_id: str,
        user_id: str,
        code: str,
        hint_level: HintLevel
    ) -> Dict[str, Any]:
        """
        Get a progressive hint for a stuck learner

        Args:
            challenge_id: Challenge being attempted
            user_id: User requesting hint
            code: Current code attempt
            hint_level: How detailed the hint should be

        Returns:
            Hint with appropriate level of detail
        """
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found")

        # Get pre-written hints if available
        hint_idx = {
            HintLevel.NUDGE: 0,
            HintLevel.GUIDANCE: 1,
            HintLevel.EXPLANATION: 2,
            HintLevel.PARTIAL: 3,
            HintLevel.SOLUTION: 4
        }.get(hint_level, 0)

        if hint_idx < len(challenge.hints):
            return {
                "hint_level": hint_level.value,
                "hint": challenge.hints[hint_idx],
                "hints_remaining": len(challenge.hints) - hint_idx - 1
            }

        # Generate AI hint if no pre-written hints available
        level_instructions = {
            HintLevel.NUDGE: "Give a very subtle hint that points in the right direction without giving anything away.",
            HintLevel.GUIDANCE: "Provide more specific guidance about the approach to take.",
            HintLevel.EXPLANATION: "Explain the concept or algorithm needed to solve this.",
            HintLevel.PARTIAL: "Show a partial solution or pseudocode.",
            HintLevel.SOLUTION: "Provide the complete solution with explanation."
        }

        prompt = f"""
CHALLENGE: {challenge.title}
DESCRIPTION: {challenge.description}

LEARNER'S CURRENT CODE:
```{challenge.language}
{code}
```

{level_instructions[hint_level]}

The goal is to help them learn, not just give answers.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a helpful coding tutor providing progressive hints."),
            HumanMessage(content=prompt)
        ])

        return {
            "hint_level": hint_level.value,
            "hint": response.content,
            "hints_remaining": 0
        }

    def get_challenge(self, challenge_id: str) -> Optional[CodingChallenge]:
        """Get challenge by ID"""
        return self.challenges.get(challenge_id)


# Singleton instance
_code_evaluator: Optional[CodeEvaluator] = None


def get_code_evaluator() -> CodeEvaluator:
    """Get or create the code evaluator singleton"""
    global _code_evaluator
    if _code_evaluator is None:
        _code_evaluator = CodeEvaluator()
    return _code_evaluator


# Pre-built challenges for common topics
def register_sample_challenges(evaluator: CodeEvaluator) -> None:
    """Register sample coding challenges"""

    # Two Sum challenge
    evaluator.create_challenge(
        challenge_id="two_sum",
        title="Two Sum",
        description="""Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].""",
        difficulty=DifficultyLevel.BEGINNER,
        function_name="two_sum",
        parameters=[
            {"name": "nums", "type": "List[int]"},
            {"name": "target", "type": "int"}
        ],
        return_type="List[int]",
        test_cases=[
            {"input": ([2, 7, 11, 15], 9), "expected": [0, 1], "description": "Basic case"},
            {"input": ([3, 2, 4], 6), "expected": [1, 2], "description": "Numbers not at start"},
            {"input": ([3, 3], 6), "expected": [0, 1], "description": "Duplicate values"},
        ],
        concepts_tested=["arrays", "hash tables", "iteration"],
        hints=[
            "Think about what information you need to track as you iterate.",
            "A hash table can help you look up values quickly.",
            "For each number, check if target - number exists in your hash table.",
        ],
        reference_solution="""def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []"""
    )

    # Palindrome check
    evaluator.create_challenge(
        challenge_id="is_palindrome",
        title="Palindrome Check",
        description="""Given a string `s`, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

Example:
Input: s = "A man, a plan, a canal: Panama"
Output: True
Explanation: "amanaplanacanalpanama" is a palindrome.""",
        difficulty=DifficultyLevel.BEGINNER,
        function_name="is_palindrome",
        parameters=[{"name": "s", "type": "str"}],
        return_type="bool",
        test_cases=[
            {"input": "A man, a plan, a canal: Panama", "expected": True, "description": "Classic palindrome"},
            {"input": "race a car", "expected": False, "description": "Not a palindrome"},
            {"input": " ", "expected": True, "description": "Empty/whitespace"},
        ],
        concepts_tested=["strings", "two pointers"],
        hints=[
            "Consider filtering out non-alphanumeric characters first.",
            "Compare characters from both ends moving inward.",
            "Two pointers can solve this efficiently."
        ],
        reference_solution="""def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]"""
    )
