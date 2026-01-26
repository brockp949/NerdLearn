from typing import Dict, Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
import json

from tests.swarm.core.fuel_meter import FuelMeter, FuelLimit, FuelType
from tests.swarm.core.antigravity_prompts import AntigravityPrompt, GoalVector, GravitationalWell

class RefinerAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.fuel = FuelMeter(limit=FuelLimit.MEDIUM, name="RefinerAgent")
        self.llm = llm
        self.goal = GoalVector(
            primary_objective="Verify Learning Outcome Alignment",
            success_criteria=[
                "Outcomes must be measurable (Bloom's Taxonomy)",
                "Outcomes must map to specific lesson content",
                "No vague verbs like 'understand' or 'know'"
            ],
            failure_conditions=[
                "Outcome is unmeasurable",
                "Outcome references missing content",
                "Hallucinated concepts not in source material"
            ]
        )
        self.gravity = GravitationalWell(intensity="HIGH")
        self.prompt_engine = AntigravityPrompt(self.goal, self.gravity)

    def verify_outcomes(self, curriculum_segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a curriculum segment and verifies its learning outcomes.
        Returns a verification report.
        """
        self.fuel.spend(50, FuelType.STEP) # Initial setup cost
        
        # Construct the antigravity prompt
        input_data = str(curriculum_segment)
        prompt = self.prompt_engine.construct(input_data)
        
        issues = []
        
        if self.llm:
            # Live Mode
            self.fuel.spend(500, FuelType.TOKEN) # Estimated cost
            try:
                response = self.llm.invoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content="Verify the provided learning outcomes. Return a JSON object with 'passed' (bool) and 'issues' (list of strings).")
                ])
                
                content = response.content
                if isinstance(content, str):
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    data = json.loads(content)
                    issues = data.get("issues", [])
                    passed = data.get("passed", False)
                else: 
                     # Handle non-string content if needed (multimodal)
                     issues.append("LLM returned non-string content")
                     passed = False
                
            except Exception as e:
                issues.append(f"LLM Error: {str(e)}")
                passed = False
        else:
            # Mock Mode
            print("--- REFINER AGENT PROMPT (MOCK) ---")
            print(prompt)
            print("----------------------------")
            
            outcomes = curriculum_segment.get("learning_outcomes", [])
            for outcome in outcomes:
                if "understand" in outcome.lower():
                    issues.append(f"Vague verb detected: '{outcome}'. Use 'Explain' or 'Demonstrate' instead.")
            
            passed = len(issues) == 0

        return {
            "agent": "Refiner",
            "passed": passed,
            "fuel_remaining": self.fuel.check_remaining(),
            "issues": issues,
            "mode": "LIVE" if self.llm else "MOCK"
        }

if __name__ == "__main__":
    print("Running Refiner in MOCK mode...")
    agent = RefinerAgent()
    sample_data = {
        "title": "Intro to Python",
        "learning_outcomes": [
            "Understand variables", 
            "Write a for loop"
        ]
    }
    result = agent.verify_outcomes(sample_data)
    print("Verification Result:", result)
