"""
TDD Challenge Generator - Test-Driven Coding Challenge Generation

Research alignment:
- Test-Driven Quality Assurance: Generate challenges with verified solutions
- Generative Assessment: Create novel coding problems dynamically
- Pyodide Integration: Browser-based execution for security and speed

Key Features:
1. LLM generates problem description
2. LLM generates unit test suite (hidden from user)
3. LLM generates reference solution
4. Verify solution passes all tests before presenting
5. Support for multiple programming languages
6. Difficulty calibration based on learner level
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import json
import re
import hashlib

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProgrammingLanguage(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    SQL = "sql"


class DifficultyLevel(str, Enum):
    """Challenge difficulty levels"""
    BEGINNER = "beginner"       # Basic syntax, simple logic
    EASY = "easy"               # Standard problems, single function
    MEDIUM = "medium"           # Multiple functions, data structures
    HARD = "hard"               # Complex algorithms, optimization
    EXPERT = "expert"           # Advanced patterns, system design


class ChallengeCategory(str, Enum):
    """Categories of coding challenges"""
    ALGORITHMS = "algorithms"
    DATA_STRUCTURES = "data_structures"
    STRING_MANIPULATION = "string_manipulation"
    ARRAYS = "arrays"
    RECURSION = "recursion"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    OBJECT_ORIENTED = "object_oriented"
    FUNCTIONAL = "functional"
    DATABASE = "database"
    API_DESIGN = "api_design"


@dataclass
class TestCase:
    """Individual test case"""
    name: str
    input_data: Any
    expected_output: Any
    is_hidden: bool = False  # Hidden tests not shown to user
    description: str = ""


@dataclass
class CodingChallenge:
    """Complete coding challenge"""
    id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    category: ChallengeCategory
    language: ProgrammingLanguage

    # Problem specification
    function_signature: str
    constraints: List[str]
    examples: List[Dict[str, Any]]

    # Solution and tests
    test_cases: List[TestCase]
    reference_solution: str
    starter_code: str

    # Metadata
    estimated_minutes: int
    concepts_tested: List[str]
    hints: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Quality flags
    verified: bool = False  # True if reference solution passes all tests
    quality_score: float = 0.0

    def to_dict(self, include_solution: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding solution"""
        data = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "category": self.category.value,
            "language": self.language.value,
            "function_signature": self.function_signature,
            "constraints": self.constraints,
            "examples": self.examples,
            "starter_code": self.starter_code,
            "estimated_minutes": self.estimated_minutes,
            "concepts_tested": self.concepts_tested,
            "hints": self.hints,
            "test_cases": [
                {
                    "name": tc.name,
                    "input": tc.input_data,
                    "expected": tc.expected_output,
                    "description": tc.description
                }
                for tc in self.test_cases if not tc.is_hidden
            ],
            "verified": self.verified,
            "quality_score": self.quality_score
        }

        if include_solution:
            data["reference_solution"] = self.reference_solution
            data["hidden_test_cases"] = [
                {
                    "name": tc.name,
                    "input": tc.input_data,
                    "expected": tc.expected_output
                }
                for tc in self.test_cases if tc.is_hidden
            ]

        return data


@dataclass
class ExecutionResult:
    """Result of code execution"""
    passed: bool
    test_results: List[Dict[str, Any]]
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    stdout: str = ""


class TDDChallengeGenerator:
    """
    Test-Driven Development Challenge Generator

    Generates coding challenges with guaranteed-working solutions:

    1. Problem Generation: LLM creates problem description and constraints
    2. Test Generation: LLM creates comprehensive test suite
    3. Solution Generation: LLM creates reference solution
    4. Verification: Execute solution against tests to verify correctness
    5. Only present challenge if verification passes

    This ensures learners never receive unsolvable problems.

    Example Usage:
    ```python
    generator = TDDChallengeGenerator()

    # Generate a challenge
    challenge = await generator.generate(
        topic="Array Manipulation",
        difficulty=DifficultyLevel.MEDIUM,
        language=ProgrammingLanguage.PYTHON,
        concepts=["loops", "list comprehension"]
    )

    # Evaluate user submission
    result = await generator.evaluate_submission(
        challenge_id=challenge.id,
        user_code="def solution(arr): return sorted(arr)"
    )
    ```
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        code_executor: Optional[Any] = None
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.code_executor = code_executor  # Pyodide or other executor

        # Cache of generated challenges
        self._challenge_cache: Dict[str, CodingChallenge] = {}

        # Templates for different difficulty levels
        self._difficulty_specs = {
            DifficultyLevel.BEGINNER: {
                "test_count": (3, 5),
                "constraint_count": (1, 2),
                "hint_count": 3,
                "estimated_minutes": (5, 10)
            },
            DifficultyLevel.EASY: {
                "test_count": (4, 6),
                "constraint_count": (2, 3),
                "hint_count": 2,
                "estimated_minutes": (10, 15)
            },
            DifficultyLevel.MEDIUM: {
                "test_count": (5, 8),
                "constraint_count": (3, 4),
                "hint_count": 2,
                "estimated_minutes": (15, 25)
            },
            DifficultyLevel.HARD: {
                "test_count": (6, 10),
                "constraint_count": (3, 5),
                "hint_count": 1,
                "estimated_minutes": (25, 40)
            },
            DifficultyLevel.EXPERT: {
                "test_count": (8, 12),
                "constraint_count": (4, 6),
                "hint_count": 1,
                "estimated_minutes": (40, 60)
            }
        }

    def _generate_challenge_id(self, title: str, topic: str) -> str:
        """Generate unique challenge ID"""
        content = f"{title}:{topic}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def generate(
        self,
        topic: str,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
        concepts: Optional[List[str]] = None,
        category: Optional[ChallengeCategory] = None,
        max_retries: int = 3
    ) -> Optional[CodingChallenge]:
        """
        Generate a verified coding challenge

        Args:
            topic: Topic area for the challenge
            difficulty: Difficulty level
            language: Programming language
            concepts: Specific concepts to test
            category: Challenge category
            max_retries: Maximum generation attempts

        Returns:
            Verified CodingChallenge or None if generation failed
        """
        logger.info(f"Generating {difficulty.value} {language.value} challenge for: {topic}")

        for attempt in range(max_retries):
            try:
                # Step 1: Generate problem description
                problem = await self._generate_problem(
                    topic, difficulty, language, concepts, category
                )

                if not problem:
                    logger.warning(f"Problem generation failed (attempt {attempt + 1})")
                    continue

                # Step 2: Generate test cases
                test_cases = await self._generate_tests(
                    problem, difficulty, language
                )

                if not test_cases or len(test_cases) < 3:
                    logger.warning(f"Test generation failed (attempt {attempt + 1})")
                    continue

                # Step 3: Generate reference solution
                solution = await self._generate_solution(
                    problem, test_cases, language
                )

                if not solution:
                    logger.warning(f"Solution generation failed (attempt {attempt + 1})")
                    continue

                # Step 4: Verify solution passes tests
                verified, verification_result = await self._verify_solution(
                    solution, test_cases, language
                )

                if not verified:
                    logger.warning(f"Solution verification failed (attempt {attempt + 1}): {verification_result}")
                    continue

                # Build the challenge
                challenge_id = self._generate_challenge_id(problem["title"], topic)
                specs = self._difficulty_specs[difficulty]

                challenge = CodingChallenge(
                    id=challenge_id,
                    title=problem["title"],
                    description=problem["description"],
                    difficulty=difficulty,
                    category=category or ChallengeCategory.ALGORITHMS,
                    language=language,
                    function_signature=problem["function_signature"],
                    constraints=problem.get("constraints", []),
                    examples=problem.get("examples", []),
                    test_cases=test_cases,
                    reference_solution=solution,
                    starter_code=problem.get("starter_code", ""),
                    estimated_minutes=sum(specs["estimated_minutes"]) // 2,
                    concepts_tested=concepts or [],
                    hints=problem.get("hints", []),
                    verified=True,
                    quality_score=verification_result.get("score", 1.0)
                )

                # Cache and return
                self._challenge_cache[challenge_id] = challenge

                logger.info(f"Successfully generated challenge: {challenge.title}")
                return challenge

            except Exception as e:
                logger.error(f"Challenge generation error (attempt {attempt + 1}): {e}")

        logger.error(f"Failed to generate challenge after {max_retries} attempts")
        return None

    async def _generate_problem(
        self,
        topic: str,
        difficulty: DifficultyLevel,
        language: ProgrammingLanguage,
        concepts: Optional[List[str]],
        category: Optional[ChallengeCategory]
    ) -> Optional[Dict[str, Any]]:
        """Generate problem description"""
        specs = self._difficulty_specs[difficulty]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating coding challenges.

Create clear, well-specified problems that:
1. Have a single, unambiguous correct answer
2. Can be solved in a single function
3. Have clear input/output types
4. Include edge cases in constraints

Language-specific requirements:
- Python: Use type hints in signature
- JavaScript: Use JSDoc comments
- TypeScript: Use TypeScript types
- SQL: Specify table schema"""),
            ("human", """Create a {difficulty} coding challenge:

Topic: {topic}
Language: {language}
Category: {category}
Concepts to test: {concepts}

Generate JSON:
{{
    "title": "Brief, descriptive title",
    "description": "Full problem description with context",
    "function_signature": "Complete function signature",
    "constraints": ["List of constraints"],
    "examples": [
        {{
            "input": "Example input",
            "output": "Expected output",
            "explanation": "Why this output"
        }}
    ],
    "starter_code": "Code template with function signature",
    "hints": ["Progressive hints for stuck learners"]
}}""")
        ])

        try:
            messages = prompt.format_messages(
                difficulty=difficulty.value,
                topic=topic,
                language=language.value,
                category=category.value if category else "general",
                concepts=", ".join(concepts) if concepts else "general programming"
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            problem = json.loads(response_text.strip())
            return problem

        except Exception as e:
            logger.error(f"Error generating problem: {e}")
            return None

    async def _generate_tests(
        self,
        problem: Dict[str, Any],
        difficulty: DifficultyLevel,
        language: ProgrammingLanguage
    ) -> List[TestCase]:
        """Generate test cases for the problem"""
        specs = self._difficulty_specs[difficulty]
        min_tests, max_tests = specs["test_count"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You generate comprehensive test suites for coding challenges.

Create tests that cover:
1. Basic/happy path cases
2. Edge cases (empty input, single element, etc.)
3. Boundary conditions
4. Invalid input handling (if applicable)

Mark some tests as "hidden" to prevent solution gaming."""),
            ("human", """Generate tests for this problem:

Title: {title}
Description: {description}
Function: {signature}
Examples: {examples}

Generate {min_tests} to {max_tests} test cases as JSON:
[
    {{
        "name": "test_descriptive_name",
        "input": <input value(s)>,
        "expected": <expected output>,
        "is_hidden": false,
        "description": "What this test verifies"
    }}
]

Include at least 2 hidden tests.""")
        ])

        try:
            messages = prompt.format_messages(
                title=problem["title"],
                description=problem["description"][:500],
                signature=problem["function_signature"],
                examples=json.dumps(problem.get("examples", [])),
                min_tests=min_tests,
                max_tests=max_tests
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            tests_data = json.loads(response_text.strip())

            test_cases = []
            for t in tests_data:
                test = TestCase(
                    name=t.get("name", f"test_{len(test_cases)}"),
                    input_data=t.get("input"),
                    expected_output=t.get("expected"),
                    is_hidden=t.get("is_hidden", False),
                    description=t.get("description", "")
                )
                test_cases.append(test)

            return test_cases

        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return []

    async def _generate_solution(
        self,
        problem: Dict[str, Any],
        test_cases: List[TestCase],
        language: ProgrammingLanguage
    ) -> Optional[str]:
        """Generate reference solution"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You write optimal, clean solutions to coding problems.

Requirements:
1. Code must be correct and handle all edge cases
2. Use idiomatic {language} patterns
3. Include brief comments for complex logic
4. Optimize for readability, then performance"""),
            ("human", """Write a solution for:

Title: {title}
Description: {description}
Signature: {signature}
Test cases: {test_cases}

Write only the function code, no explanations:""")
        ])

        try:
            # Format test cases for context
            test_summary = [
                {"input": tc.input_data, "expected": tc.expected_output}
                for tc in test_cases[:5]  # Limit to avoid token overflow
            ]

            messages = prompt.format_messages(
                language=language.value,
                title=problem["title"],
                description=problem["description"][:500],
                signature=problem["function_signature"],
                test_cases=json.dumps(test_summary)
            )

            response = await self.llm.ainvoke(messages)

            # Extract code
            solution = response.content
            if "```" in solution:
                # Extract code from markdown code block
                match = re.search(r'```(?:\w+)?\n(.*?)```', solution, re.DOTALL)
                if match:
                    solution = match.group(1)

            return solution.strip()

        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return None

    async def _verify_solution(
        self,
        solution: str,
        test_cases: List[TestCase],
        language: ProgrammingLanguage
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify solution passes all test cases

        In production, this would execute against Pyodide or a sandboxed executor.
        For now, we use LLM-based verification as a fallback.
        """
        # If we have a code executor, use it
        if self.code_executor:
            try:
                result = await self.code_executor.run(
                    solution,
                    test_cases,
                    language
                )
                return (result.passed, {"score": 1.0 if result.passed else 0.0})
            except Exception as e:
                logger.warning(f"Code executor failed, falling back to LLM verification: {e}")

        # Fallback: LLM-based verification
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You verify if code solutions are correct.

Execute each test case mentally and verify the output.
Be strict - the solution must be correct for ALL tests."""),
            ("human", """Verify this solution:

```{language}
{solution}
```

Test cases:
{test_cases}

For each test, mentally execute and verify.

Output JSON:
{{
    "all_passed": true/false,
    "results": [
        {{"test": "name", "passed": true/false, "actual": <actual output if different>}}
    ],
    "issues": ["List any issues found"]
}}""")
        ])

        try:
            test_data = [
                {"name": tc.name, "input": tc.input_data, "expected": tc.expected_output}
                for tc in test_cases
            ]

            messages = prompt.format_messages(
                language=language.value,
                solution=solution,
                test_cases=json.dumps(test_data)
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            verification = json.loads(response_text.strip())

            passed = verification.get("all_passed", False)
            results = verification.get("results", [])

            passed_count = sum(1 for r in results if r.get("passed", False))
            score = passed_count / len(test_cases) if test_cases else 0.0

            return (passed, {"score": score, "results": results})

        except Exception as e:
            logger.error(f"Error verifying solution: {e}")
            return (False, {"score": 0.0, "error": str(e)})

    async def evaluate_submission(
        self,
        challenge_id: str,
        user_code: str
    ) -> ExecutionResult:
        """
        Evaluate a user's code submission

        Args:
            challenge_id: The challenge ID
            user_code: User's submitted code

        Returns:
            ExecutionResult with test results
        """
        challenge = self._challenge_cache.get(challenge_id)
        if not challenge:
            return ExecutionResult(
                passed=False,
                test_results=[],
                error_message="Challenge not found"
            )

        # Run against all test cases (including hidden)
        passed, result = await self._verify_solution(
            user_code,
            challenge.test_cases,
            challenge.language
        )

        # Format results
        test_results = []
        for tc in challenge.test_cases:
            # Find matching result
            matching = next(
                (r for r in result.get("results", []) if r.get("test") == tc.name),
                None
            )

            test_results.append({
                "name": tc.name,
                "passed": matching.get("passed", False) if matching else False,
                "input": tc.input_data if not tc.is_hidden else "<hidden>",
                "expected": tc.expected_output if not tc.is_hidden else "<hidden>",
                "actual": matching.get("actual") if matching else None,
                "is_hidden": tc.is_hidden
            })

        return ExecutionResult(
            passed=passed,
            test_results=test_results,
            error_message=result.get("error"),
            execution_time_ms=0.0  # Would be filled by actual executor
        )

    def get_challenge(self, challenge_id: str) -> Optional[CodingChallenge]:
        """Retrieve a cached challenge"""
        return self._challenge_cache.get(challenge_id)

    async def generate_hint(
        self,
        challenge_id: str,
        user_code: str,
        hint_level: int = 1
    ) -> str:
        """
        Generate a contextual hint based on user's current code

        Args:
            challenge_id: Challenge ID
            user_code: User's current code
            hint_level: 1=subtle, 2=moderate, 3=explicit

        Returns:
            Hint string
        """
        challenge = self._challenge_cache.get(challenge_id)
        if not challenge:
            return "Challenge not found"

        hint_levels = {
            1: "Give a subtle hint that points in the right direction without revealing the approach",
            2: "Give a moderate hint that suggests the general approach",
            3: "Give an explicit hint that describes the algorithm but not the code"
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You provide helpful hints for coding challenges.

Hint guidelines:
- Level 1 (subtle): Point toward the right concept or data structure
- Level 2 (moderate): Suggest the general approach
- Level 3 (explicit): Describe the algorithm without code

Never provide actual code in hints."""),
            ("human", """Challenge: {title}
Description: {description}

User's current code:
```
{user_code}
```

{hint_instruction}

Provide the hint:""")
        ])

        try:
            messages = prompt.format_messages(
                title=challenge.title,
                description=challenge.description[:300],
                user_code=user_code or "// No code yet",
                hint_instruction=hint_levels.get(hint_level, hint_levels[2])
            )

            response = await self.llm.ainvoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return "Think about the problem step by step."


# Lazy-initialized singleton
_tdd_generator: Optional[TDDChallengeGenerator] = None


def get_tdd_generator() -> TDDChallengeGenerator:
    """Get or create the TDD challenge generator"""
    global _tdd_generator
    if _tdd_generator is None:
        _tdd_generator = TDDChallengeGenerator()
    return _tdd_generator
