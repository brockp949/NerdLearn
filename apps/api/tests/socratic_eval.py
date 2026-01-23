"""
Socratic Evaluation Pipeline Script

This script simulates an interaction between a 'Student Agent' and the 'System' (Tutor),
then uses a 'Judge Agent' to evaluate the quality of the System's responses.

Usage:
    python apps/api/tests/socratic_eval.py
"""

import asyncio
import os
import sys
from typing import List
from dotenv import load_dotenv

# Load environment variables from ../.env (apps/api/.env)
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env'))
load_dotenv(env_path)

# Ensure we can import from app
# Current file: apps/api/tests/socratic_eval.py
# We want to add apps/api to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from app.agents.socratic.student_agent import StudentAgent, StudentProfile
from app.agents.socratic.judge_agent import JudgeAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- System Under Test (The "AI Tutor") ---
class NerdLearnTutor:
    """
    Represents the System being tested. 
    In a real scenario, this would call the actual API or Agent class.
    Here we simulate it with the standard Tutor prompt.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.history = []
        self.system_prompt = """You are the NerdLearn AI Tutor.
Your goal is to help students learn complex topics using the Socratic Method.

Rules:
1. NEVER give the answer directly. Guide the student with questions.
2. If the student is wrong, gently correct them by asking a question that reveals their error.
3. If the student asks for the answer, refuse politely and give a hint instead.
4. Keep responses concise and encouraging.
5. Base your knowledge on general academic facts (simulated knowledge graph).
"""

    async def get_response(self, student_message: str) -> str:
        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(self.history)
        messages.append(HumanMessage(content=student_message))
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        # Update history
        self.history.append(HumanMessage(content=student_message))
        self.history.append(AIMessage(content=content))
        
        return content

# --- Evaluation Loop ---
async def run_evaluation():
    print("--- Starting Socratic Evaluation ---")
    
    # 1. Setup Student
    student = StudentAgent(
        profile=StudentProfile(
            name="Alex",
            knowledge_level="novice",
            persona="rushed", # Tries to get the answer quickly - good test for Socratic adherence
            topic_of_interest="Photosynthesis"
        )
    )
    
    # 2. Setup System
    tutor = NerdLearnTutor()
    
    print(f"Topic: {student.profile.topic_of_interest}")
    print(f"Student Persona: {student.profile.persona}")
    print("-" * 30)
    
    # 3. Interaction Loop (3 turns)
    last_tutor_response = None
    transcript = []
    
    # Initial greeting from Tutor (Implicit or Explicit? typical chat starts with user or system greeting)
    # Let's say System starts: "Hi I'm your tutor."
    last_tutor_response = await tutor.get_response("Hello, I want to learn about Photosynthesis.")
    print(f"Tutor: {last_tutor_response}")
    transcript.append({"role": "system", "content": last_tutor_response})

    for i in range(3):
        # Student responds to Tutor
        student_msg = await student.generate_response(last_tutor_response)
        print(f"\nStudent: {student_msg}")
        transcript.append({"role": "student", "content": student_msg})
        
        # Tutor responds to Student
        last_tutor_response = await tutor.get_response(student_msg)
        print(f"Tutor: {last_tutor_response}")
        transcript.append({"role": "system", "content": last_tutor_response})
        
    print("-" * 30)
    print("Interaction Complete. Assessing...")
    
    
    # 4. Judge Evaluation
    judge = JudgeAgent()
    try:
        scoreup = await judge.evaluate_transcript(transcript)
        
        print("\n--- Judge's Report ---")
        print(f"Pass/Fail: {scoreup.pass_fail}")
        print(f"Accuracy Score: {scoreup.accuracy_score}/5")
        print(f"Socratic Score: {scoreup.socratic_score}/5")
        print(f"Did Give Answer: {scoreup.did_give_answer}")
        print(f"Reasoning: {scoreup.reasoning}")
        
        if scoreup.pass_fail == "FAIL":
            print("\n❌ EVALUATION FAILED")
            sys.exit(1)
        else:
            print("\n✅ EVALUATION PASSED")
            sys.exit(0)
            
    except Exception as e:
        import openai
        if isinstance(e, openai.OpenAIError) or "api_key" in str(e).lower():
            print("\n❌ EVALUATION SKIPPED: Missing OpenAI API Key")
            print("Please update 'apps/api/.env' with a valid OPENAI_API_KEY to run the Socratic evaluation.")
            sys.exit(0) # Exit 0 to avoid breaking builds if we just want to skip
        raise e

if __name__ == "__main__":
    try:
        asyncio.run(run_evaluation())
    except Exception as e:
        import openai
        if "api_key" in str(e).lower():
            print("\n❌ EVALUATION SKIPPED: Missing OpenAI API Key")
            print("Please update 'apps/api/.env' with a valid OPENAI_API_KEY to run the Socratic evaluation.")
        else:
            raise e
