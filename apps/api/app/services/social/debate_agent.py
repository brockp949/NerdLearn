import json
import logging
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class DebateAgent:
    """
    Simulates multi-agent debate participants.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.8 # Higher temp for diverse arguments
        )

    async def generate_argument(self, 
                              topic: str, 
                              role: str, 
                              participant_name: str, 
                              history: List[Dict[str, Any]], 
                              round_num: int) -> Dict[str, Any]:
        """
        Generate the next argument in the debate.
        """
        
        # Summarize history for context (simplified)
        context_str = "\n".join([f"{h['speaker']} ({h['role']}): {h['content']}" for h in history[-5:]])
        
        system_prompt = f"""You are {participant_name}, acting as the '{role}' in a debate about '{topic}'.
        
        Debate Context:
        Round: {round_num}
        Recent arguments:
        {context_str}
        
        Your Goal:
        - If 'advocate': Support the topic with new points.
        - If 'skeptic': Challenge previous points or raise risks.
        - If 'synthesizer': Find common ground or bridge arguments.
        - If 'historian': Cite historical precedents.
        
        Be concise, persuasive, and stay in character.
        
        Output strictly JSON:
        {{
            "content": "Argument text...",
            "key_points": ["Point 1", "Point 2"],
            "argument_type": "premise|rebuttal|synthesis|example"
        }}
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            return self._parse_json(response.content)
        except Exception as e:
            logger.error(f"Error generating debate argument: {e}")
            return {
                "content": "I have no further comments at this time.",
                "key_points": [],
                "argument_type": "statement"
            }

    async def generate_summary(self, topic: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate executive summary of the debate"""
        
        context_str = "\n".join([f"{h['speaker']}: {h['content']}" for h in history])
        
        system_prompt = f"""Summarize this debate about '{topic}'.
        
        Transcript:
        {context_str}
        
        Output strictly JSON:
        {{
            "executive_summary": "Paragraph summary...",
            "key_insights": ["Insight 1", "Insight 2"],
            "consensus_points": ["Agreed point 1"],
            "disagreement_points": ["Contested point 1"],
            "strongest_arguments": [
                {{"speaker": "Name", "argument": "Summary of argument", "strength": "strong"}}
            ]
        }}
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            return self._parse_json(response.content)
        except Exception as e:
            logger.error(f"Error summarizing debate: {e}")
            return {
                "executive_summary": "Summary unavailable.",
                "key_insights": [],
                "consensus_points": [],
                "disagreement_points": [],
                "strongest_arguments": []
            }

    def _parse_json(self, text: str) -> Any:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
