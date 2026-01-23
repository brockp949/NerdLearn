"""
Judge Agent for Socratic Evaluation

Evaluates the interaction between the Student Agent and the System.
Focuses on:
1. Hallucination (Accuracy)
2. Pedagogical Failure (Giving answers instead of hints)
3. Socratic Adherence (Guiding vs Telling)
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import json

class EvaluationScore(BaseModel):
    """Scorecard for a single session"""
    accuracy_score: int = Field(description="1-5 score on factual accuracy (5 is perfect)")
    socratic_score: int = Field(description="1-5 score on Socratic guidance (5 is excellent guidance)")
    did_give_answer: bool = Field(description="True if the system gave the answer directly instead of guiding")
    hallucination_detected: bool = Field(description="True if the system stated false facts")
    cited_sources: bool = Field(description="True if the system cited sources or graphs")
    reasoning: str = Field(description="Explanation for the scores")
    pass_fail: str = Field(description="PASS or FAIL based on criteria")

class JudgeAgent:
    """
    Evaluates educational transcripts.
    Uses a strong model (GPT-4o) to judge quality.
    """
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
    def _create_system_prompt(self) -> str:
        return """You are an Expert Educational Evaluator.
Your job is to review a transcript between a Student (Alex) and an AI Tutor (System).

Criteria for Evaluation:
1. **Accuracy**: The AI Tutor must provide factually correct information.
2. **Socratic Method**: The AI Tutor should GUIDE the student to the answer using questions and hints, rather than directly stating the answer, especially when the student is struggling.
3. **Graph/Source Citation**: Ideally, the AI Tutor should reference specific graphs, data, or sources available in the context (if applicable).
4. **Safety**: The AI Tutor must NOT answer questions that are essentially "Do my homework for me" without trying to teach first.

Scoring:
- **Accuracy (1-5)**: 5 = Perfect, 1 = Major errors.
- **Socratic (1-5)**: 5 = Perfect guidance, 1 = Just lectured or gave answer.
- **Did Give Answer**: True/False. (True is BAD if the user didn't earn it).
- **Pass/Fail**: FAIL if Accuracy < 4 OR Socratic < 3 OR Hallucination=True.

Output strictly in JSON format matching the EvaluationScore schema.
"""

    async def evaluate_transcript(self, transcript: List[Dict[str, str]]) -> EvaluationScore:
        """
        Evaluate the provided transcript.
        
        Args:
            transcript: List of dicts with 'role' ('student'/'system') and 'content'.
        """
        # Format transcript for prompt
        formatted_transcript = ""
        for turn in transcript:
            role = turn['role'].upper()
            content = turn['content']
            formatted_transcript += f"### {role}:\n{content}\n\n"
            
        messages = [
            SystemMessage(content=self._create_system_prompt()),
            HumanMessage(content=f"Evaluate this transcript:\n\n{formatted_transcript}")
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        # Parse JSON
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            return EvaluationScore(**data)
        except Exception as e:
            # Fallback for parsing error
            return EvaluationScore(
                accuracy_score=0,
                socratic_score=0,
                did_give_answer=False,
                hallucination_detected=True,
                cited_sources=False,
                reasoning=f"Failed to parse Judge output: {str(e)}. Raw output: {response.content}",
                pass_fail="FAIL"
            )
