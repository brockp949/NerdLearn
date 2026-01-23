import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class TeachableAgent:
    """
    Implements the 'Feynman Technique' partner.
    Acts as a novice student that the user must teach.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.7
        )

    async def start_session(self, concept: str, persona: str) -> Dict[str, Any]:
        """Generate the opening question/statement based on persona"""
        
        system_prompt = f"""You are a student with the persona: {persona}.
        You are about to be taught the concept: '{concept}'.
        You know very little about this topic initially.
        
        Generate a short, natural opening greeting and a question to kick off the lesson.
        Be consistent with your persona (e.g., if 'curious', be enthusiastic; if 'skeptic', be doubtful).
        
        Output strictly JSON:
        {{
            "greeting": "Hi...",
            "opening_question": "So, what exactly is..."
        }}
        """
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            data = self._parse_json(response.content)
            return data
        except Exception as e:
            logger.error(f"Error starting teaching session: {e}")
            return {
                "greeting": "Hello!",
                "opening_question": f"Can you explain {concept} to me?"
            }

    async def process_explanation(self, 
                                concept: str, 
                                persona: str, 
                                current_comprehension: float,
                                history: List[Dict[str, str]], 
                                explanation: str) -> Dict[str, Any]:
        """
        React to the user's explanation.
        Evaluate how well the user explained it and update own comprehension.
        """
        
        # Format history for context
        messages = [
            SystemMessage(content=f"""You are a student ({persona}) learning '{concept}'.
            Your current comprehension checks out at {current_comprehension:.2f}/1.0.
            
            Analyze the teacher's (user's) latest explanation.
            
            1. Did they explain clearly?
            2. Did they use jargon without defining it? (If so, ask for clarification)
            3. Did they use an analogy? (If so, praise it or ask for one)
            
            Update your internal state.
            
            Output strictly JSON:
            {{
                "response": "Your natural conversational response as the student.",
                "question_type": "clarification" | "example" | "deep_dive" | "confirmation",
                "comprehension_delta": 0.1, // float between -0.1 (confused) and 0.3 (enlightened)
                "concepts_understood": ["list", "of", "newly", "grasped", "sub-concepts"],
                "knowledge_gaps": ["list", "of", "things", "still", "unclear"]
            }}
            """)
        ]
        
        # Add limited history (last 3 exchanges) to save context window
        for msg in history[-6:]:
            if msg["role"] == "user": # User is the teacher
                messages.append(SystemMessage(content=f"Teacher said partially: {msg['content']}"))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        # Add latest input
        messages.append(HumanMessage(content=explanation))
        
        try:
            response = await self.llm.ainvoke(messages)
            return self._parse_json(response.content)
        except Exception as e:
            logger.error(f"Error processing explanation: {e}")
            return {
                "response": "I see. Could you elaborate on that?",
                "question_type": "clarification",
                "comprehension_delta": 0.05,
                "concepts_understood": [],
                "knowledge_gaps": ["Details"]
            }

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback for malformed generation
            return {"response": text, "question_type": "general", "comprehension_delta": 0.0}
