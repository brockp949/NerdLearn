import pytest
from unittest.mock import MagicMock
from app.core.telemetry import TelemetryService
from app.core.cost_tracker import CostTrackingCallback
from langchain_core.outputs import LLMResult

class TestInfrastructure:
    
    def setup_method(self):
        # Reset singleton state for tests
        TelemetryService().reset_metrics()

    def test_telemetry_singleton(self):
        t1 = TelemetryService()
        t2 = TelemetryService()
        assert t1 is t2
        
        t1.track_llm_call("test-model", 100, 0.01)
        metrics = t2.get_metrics()
        assert metrics["llm_calls_total"] == 1
        assert metrics["llm_tokens_total"] == 100

    def test_cost_calculation(self):
        cb = CostTrackingCallback()
        
        # Simulate LLM Result
        llm_result = LLMResult(
            generations=[],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 1000,
                    "total_tokens": 2000
                },
                "model_name": "gpt-4o"
            }
        )
        
        cb.on_llm_end(llm_result, run_id=1)
        
        metrics = TelemetryService().get_metrics()
        assert metrics["llm_calls_total"] == 1
        assert metrics["llm_tokens_total"] == 2000
        
        # Expected cost: 
        # Input: 1000/1000 * 0.005 = 0.005
        # Output: 1000/1000 * 0.015 = 0.015
        # Total: 0.020
        assert abs(metrics["llm_cost_est_usd"] - 0.020) < 0.0001

    def test_agent_action_logging(self):
        t = TelemetryService()
        t.log_agent_action("Architect", "Create Syllabus", {"topic": "Math"})
        
        actions = t.get_recent_actions()
        assert len(actions) == 1
        assert actions[0]["agent"] == "Architect"
        assert actions[0]["metadata"]["topic"] == "Math"
