from typing import Dict, Any, List, Set, Tuple, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
import json

from tests.swarm.core.fuel_meter import FuelMeter, FuelLimit, FuelType
from tests.swarm.core.antigravity_prompts import AntigravityPrompt, GoalVector, GravitationalWell

class ArchitectAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.fuel = FuelMeter(limit=FuelLimit.HIGH, name="ArchitectAgent")
        self.llm = llm
        self.goal = GoalVector(
            primary_objective="Verify Topological Continuity",
            success_criteria=[
                "Graph must be a DAG (Directed Acyclic Graph)",
                "No orphaned nodes (except Root)",
                "Dependencies must exist in the graph"
            ],
            failure_conditions=[
                "Cycle detected in dependency tree",
                "Node reachable only via non-existent dependency",
                "Islands of disconnected nodes"
            ]
        )
        self.gravity = GravitationalWell(intensity="MAXIMUM")
        self.prompt_engine = AntigravityPrompt(self.goal, self.gravity)

    def verify_topology(self, curriculum_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the curriculum structure forms a valid DAG.
        """
        self.fuel.spend(100, FuelType.STEP)
        
        nodes = curriculum_graph.get("nodes", [])
        edges = curriculum_graph.get("edges", []) 
        
        # 1. Structural Check
        cycles = self._detect_cycles(nodes, edges)
        orphans = self._find_orphans(nodes, edges)
        
        semantic_issues = []
        prompt = self.prompt_engine.construct(str(curriculum_graph))
        
        if self.llm:
             # Live Mode
            self.fuel.spend(1000, FuelType.TOKEN)
            try:
                response = self.llm.invoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content="Analyze the dependency graph for logical fallacies. Return JSON with 'semantic_issues' (list).")
                ])
                
                content = response.content
                if isinstance(content, str):
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    data = json.loads(content)
                    semantic_issues = data.get("semantic_issues", [])
                
            except Exception as e:
                semantic_issues.append(f"LLM Error: {str(e)}")
        
        passed = (len(cycles) == 0) and (len(orphans) == 0) and (len(semantic_issues) == 0)
        
        return {
            "agent": "Architect",
            "passed": passed,
            "cycles_found": cycles,
            "orphans_found": orphans,
            "semantic_issues": semantic_issues,
            "mode": "LIVE" if self.llm else "MOCK"
        }

    def _detect_cycles(self, nodes: List[Dict], edges: List[Dict]) -> List[List[str]]:
        adj = {n["id"]: [] for n in nodes}
        for e in edges:
            if e["from"] in adj:
                adj[e["from"]].append(e["to"])
        
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    cycles.append(list(path)) 
            
            rec_stack.remove(node)
            path.pop()

        for node in nodes:
            if node["id"] not in visited:
                dfs(node["id"], [])
                
        return cycles

    def _find_orphans(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        targets = set(e["to"] for e in edges)
        ids = set(n["id"] for n in nodes)
        orphans = [n_id for n_id in ids if n_id not in targets]
        if len(orphans) > 1:
            return orphans 
        return []

if __name__ == "__main__":
    print("Running Architect in MOCK mode...")
    agent = ArchitectAgent()
    bad_graph = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edges": [{"from": "A", "to": "B"}, {"from": "B", "to": "C"}, {"from": "C", "to": "A"}]
    }
    print("Testing Bad Graph:", agent.verify_topology(bad_graph))
