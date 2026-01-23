import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.schemas.social_agent import CodingChallenge, DifficultyLevel, TestCase

logger = logging.getLogger(__name__)

class CodingAgent:
    """
    AI Agent that acts as a Code Interviewer / Challenge Creator.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.4 # Lower temp for code tasks
        )

    async def generate_challenges(self, count: int = 3) -> List[Dict[str, Any]]:
        """Generate a list of daily coding challenges"""
        
        system_prompt = f"""Generate {count} unique coding challenges for a learning platform.
        Focus on algorithmic thinking and data structures.
        Range from Beginner to Advanced.
        
        Output strictly JSON list of objects matching this schema:
        {{
            "challenge_id": "unique_id",
            "title": "Title",
            "description": "Problem statement...",
            "difficulty": "beginner|intermediate|advanced|expert",
            "language": "python",
            "function_name": "function_to_implement",
            "parameters": [{{"name": "arg", "type": "int"}}],
            "return_type": "int",
            "test_cases": [
                {{"input": [1, 2], "expected": 3, "description": "1+2=3"}}
            ],
            "concepts_tested": ["Math", "Arrays"],
            "estimated_minutes": 15
        }}
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            data = self._parse_json(response.content)
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.error(f"Error generating challenges: {e}")
            return [] # Fallback to empty list (service will fallback to mocks)

    async def evaluate_submission(self, challenge: CodingChallenge, code: str) -> Dict[str, Any]:
        """
        Evaluate code using LLM as a static analyzer / judge.
        In production, this would be paired with actual execution (Pyodide/Sandbox).
        """
        
        system_prompt = f"""You are a Senior Software Engineer evaluating a coding submission.
        
        Challenge: {challenge.title}
        Description: {challenge.description}
        Expected Function: {challenge.function_name}
        
        User Code:
        ```python
        {code}
        ```
        
        Analyze the code for:
        1. Correctness (logic check)
        2. Efficiency (Big O)
        3. Code Style (naming, comments)
        4. Edge Case handling
        
        Output strictly JSON:
        {{
            "passed": boolean,
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
            return self._parse_json(response.content)
        except Exception as e:
            logger.error(f"Error evaluating code: {e}")
            # Fail gracefully
            return {
                "passed": False, 
                "overall_score": 0, 
                "feedback": [{"type": "issue", "message": "Evaluation failed due to system error."}]
            }

    def _parse_json(self, text: str) -> Any:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
