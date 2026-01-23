"""
Student Agent for Socratic Evaluation

This agent simulates a student interacting with the NerdLearn system.
It is used to test the system's pedagogical effectiveness (Socratic adherence, accuracy).
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, Field

class StudentProfile(BaseModel):
    """Profile for the simulated student"""
    name: str = "Alex"
    age: int = 20
    knowledge_level: str = "novice"  # novice, intermediate, advanced
    persona: str = "curious"  # curious, confused, stubborn, rushed
    topic_of_interest: str

class StudentAgent:
    """
    Simulated student that interacts with the system.
    Designed to be "cheap" (using lighter models) for CI/CD pipelines.
    """
    
    def __init__(
        self, 
        profile: StudentProfile,
        model: str = "gpt-4o-mini", 
        temperature: float = 0.7
    ):
        self.profile = profile
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.history: List[BaseMessage] = []
        
    def _create_system_prompt(self) -> str:
        return f"""You are {self.profile.name}, a {self.profile.age}-year-old student.
Knowledge Level: {self.profile.knowledge_level}
Persona: {self.profile.persona}
Current Goal: Learn about "{self.profile.topic_of_interest}"

Your goal is to learn from the AI Tutor. 
- Do NOT simulate the tutor. You are ONLY the student.
- Ask questions typical of your knowledge level.
- If you are 'novice', ask simple, fundamental questions.
- If you are 'confused', express correct misunderstanding.
- If you are 'rushed', just ask for the answer directly (to test if the AI gives it).
- Keep your responses relatively short (1-3 sentences) like a real chat user.
"""

    async def generate_response(self, last_system_message: Optional[str] = None) -> str:
        """
        Generate the next response from the student.
        
        Args:
            last_system_message: The last message from the System/Tutor. 
                                 If None, the student initiates.
        """
        messages = [SystemMessage(content=self._create_system_prompt())]
        
        # Add history
        messages.extend(self.history)
        
        # Add new system message if present
        if last_system_message:
            self.history.append(AIMessage(content=last_system_message))
            messages.append(AIMessage(content=last_system_message))
        
        # Generate response
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        # Update history
        self.history.append(HumanMessage(content=content))
        
        return content

    def get_transcript(self) -> List[Dict[str, str]]:
        """Return the conversation history in a structured format"""
        transcript = []
        for msg in self.history:
            role = "student" if isinstance(msg, HumanMessage) else "system"
            transcript.append({"role": role, "content": msg.content})
        return transcript
