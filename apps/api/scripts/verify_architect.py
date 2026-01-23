import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from app.agents.architect_agent import ArchitectAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_verification():
    logger.info("Starting Architect Agent Verification...")
    
    # 1. Mock Dependencies
    mock_graph_service = AsyncMock()
    mock_graph_service.find_concepts_without_prerequisites.return_value = ["Linear Algebra", "Complex Numbers"]
    mock_graph_service.find_terminal_concepts.return_value = ["Shor's Algorithm", "Quantum Teleportation"]
    mock_graph_service.get_course_graph.return_value = {
        "nodes": [{"id": "Qubits"}, {"id": "Superposition"}], 
        "edges": [{"source": "Qubits", "target": "Superposition"}]
    }
    mock_graph_service.get_graph_stats.return_value = {"concept_count": 10}

    # 2. Initialize Agent with mocked LLM via base class kwargs or patching
    # Since BaseAgent initializes ChatOpenAI, we'll patch the llm directly after init
    agent = ArchitectAgent(graph_service=mock_graph_service)
    agent.llm = AsyncMock()
    
    # Mock LLM Response
    mock_response_content = """
    ```json
    {
        "modules": [
            {
                "week": 1,
                "title": "Quantum Foundations",
                "concepts": ["Linear Algebra", "Complex Numbers", "Qubits"],
                "difficulty": 3,
                "prerequisites": [],
                "rationale": "Essential math background."
            },
            {
                "week": 2,
                "title": "Quantum Gates",
                "concepts": ["Hadamard Gate", "CNOT", "Superposition"],
                "difficulty": 5,
                "prerequisites": ["Qubits"],
                "rationale": "Building blocks of circuits."
            }
        ],
        "overall_arc": "From math basics to quantum circuits."
    }
    ```
    """
    agent.llm.ainvoke.return_value = AIMessage(content=mock_response_content)

    # 3. Define Test State
    state = {
        "topic": "Quantum Computing",
        "course_id": 1,
        "constraints": {"duration_weeks": 2, "difficulty_level": "beginner"},
        "messages": [],
        "errors": []
    }

    # 4. Run Agent Process
    logger.info("Running Architect process for topic: Quantum Computing")
    new_state = await agent.process(state)

    # 5. Verify Output
    arc = new_state.get("arc_of_learning")
    if arc and len(arc.get("modules")) == 2:
        logger.info("✅ SUCCESS: Architect generated valid curriculum structure.")
        logger.info(f"Arc Summary: {arc.get('overall_arc')}")
    else:
        logger.error("❌ FAILURE: Architect output invalid.")
        logger.error(f"State: {new_state}")

if __name__ == "__main__":
    asyncio.run(run_verification())
