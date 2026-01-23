import json
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.schemas.social_agent import CodingChallenge as SchemaChallenge
from app.agents.tdd_challenge_generator import get_tdd_generator, DifficultyLevel, ProgrammingLanguage, ChallengeCategory
from app.services.code_execution.executor import SafePythonExecutor

logger = logging.getLogger(__name__)

class CodingAgent:
    """
    AI Agent that acts as a Code Interviewer / Challenge Creator.
    Integrates TDD Generator for verified challenges and Executor for solution checking.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.4
        )
        self.tdd_generator = get_tdd_generator()
        self.executor = SafePythonExecutor(timeout=2.0)

    async def generate_challenges(self, count: int = 3) -> List[Dict[str, Any]]:
        """Generate a list of verified coding challenges"""
        challenges = []
        difficulties = [DifficultyLevel.BEGINNER, DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
        topics = ["Arrays", "Strings", "Math", "Logic", "Lists"]
        
        for i in range(count):
            diff = difficulties[i % len(difficulties)]
            topic = topics[i % len(topics)]
            
            try:
                challenge = await self.tdd_generator.generate(
                    topic=topic,
                    difficulty=diff,
                    language=ProgrammingLanguage.PYTHON
                )
                
                if challenge:
                    # Convert to dict format expected by schema
                    c_dict = challenge.to_dict()
                    # Map fields if necessary to match SchemaChallenge
                    challenges.append({
                        "challenge_id": c_dict["id"],
                        "title": c_dict["title"],
                        "description": c_dict["description"],
                        "difficulty": c_dict["difficulty"],
                        "category": c_dict["category"],
                        "concepts": c_dict["concepts_tested"],
                        "function_name": c_dict["function_signature"].split("(")[0].split(" ")[-1].strip(), # Naive parse, better to have generator provide it
                        "parameters": [], # TDD generator might not strictly separate these, leaving empty or parsing sig
                        "return_type": "Any",
                        "test_cases": c_dict["test_cases"],
                        "estimated_minutes": c_dict["estimated_minutes"],
                        "language": c_dict["language"]
                    })
            except Exception as e:
                logger.error(f"Failed to generate challenge {i}: {e}")
                
        return challenges

    async def evaluate_submission(self, challenge: Union[SchemaChallenge, Dict[str, Any]], code: str) -> Dict[str, Any]:
        """
        Evaluate code using Executor for correctness + LLM for style/quality.
        """
        # 1. Prepare Data
        if hasattr(challenge, "dict"):
            c_data = challenge.dict()
        else:
            c_data = challenge
            
        function_name = c_data.get("function_name")
        if not function_name:
            # Fallback extraction if missing
            func_sig = c_data.get("function_signature", "")
            if func_sig:
                 function_name = func_sig.split("(")[0].split(" ")[-1].strip()
            else:
                 function_name = "solution" # Default fallback
        
        # normalized test cases
        test_cases = []
        raw_tests = c_data.get("test_cases", [])
        for tc in raw_tests:
            if hasattr(tc, "dict"):
                 test_cases.append(tc.dict())
            elif isinstance(tc, dict):
                 test_cases.append(tc)
            else:
                 # Should not happen based on schema
                 pass

        # 2. Execute Code
        exec_result = self.executor.execute(code, test_cases, function_name)
        
        # 3. LLM Qualitative Analysis
        status_str = "PASSED" if exec_result.passed else "FAILED"
        exec_details = (
            f"Execution Result: {status_str}\n"
            f"Error: {exec_result.error or 'None'}\n"
            f"Time: {exec_result.execution_time:.3f}s\n"
            f"Tests Passed: {sum(1 for t in exec_result.test_results if t['passed'])}/{len(test_cases)}"
        )
        
        system_prompt = f"""You are a Senior Software Engineer evaluating a coding submission.
        
        Challenge: {c_data.get('title')}
        Description: {c_data.get('description')}
        
        User Code:
        ```python
        {code}
        ```
        
        Automated Tests:
        {exec_details}
        
        Analyze the code for:
        1. Correctness (logic check against tests)
        2. Efficiency (Big O)
        3. Code Style (naming, comments)
        4. Edge Case handling (did they miss anything tests didn't catch?)
        
        Output strictly JSON:
        {{
            "passed": {str(exec_result.passed).lower()},
            "overall_score": 0-100,
            "dimension_scores": {{
                "correctness": {{ "score": 0-100, "feedback": "...", "strengths": [], "improvements": [] }},
                "efficiency": {{ "score": 0-100, "feedback": "...", "strengths": [], "improvements": [] }},
                "style": {{ "score": 0-100, "feedback": "...", "strengths": [], "improvements": [] }}
            }},
            "feedback": [
                {{ "type": "issue|suggestion|praise", "message": "...", "line_number": 1 }}
            ],
            "concepts_demonstrated": ["Recursion", "Typing"],
            "concepts_to_review": ["Error Handling"]
        }}
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            result = self._parse_json(response.content)
            
            # 4. Merge Quantitative Results (Source of Truth)
            result["passed"] = exec_result.passed
            result["tests_passed"] = sum(1 for t in exec_result.test_results if t['passed'])
            
            if not exec_result.passed:
                # Cap score if failed automated tests
                result["overall_score"] = min(result.get("overall_score", 0), 40)
                
            # Add execution error to feedback if present
            if exec_result.error:
                result.setdefault("feedback", []).insert(0, {
                    "type": "issue",
                    "message": f"Runtime Error: {exec_result.error}",
                    "line_number": 0
                })
                
            return result
        except Exception as e:
            logger.error(f"Error evaluating code: {e}")
            return {
                "passed": exec_result.passed, 
                "overall_score": 0, 
                "feedback": [{"type": "issue", "message": "Evaluation failed due to system error."}]
            }

    def _parse_json(self, text: str) -> Any:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
