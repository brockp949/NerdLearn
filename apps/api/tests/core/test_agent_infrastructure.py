import pytest
from app.agents.base_agent import BaseAgent
from app.core.telemetry import TelemetryService
from unittest.mock import MagicMock, patch

class TestAgentInfrastructure:
    
    def test_agent_has_cost_callback(self):
        # We need to mock ChatOpenAI so we don't actually need an API key
        with patch('app.agents.base_agent.ChatOpenAI') as MockChat:
            agent = BaseAgent(name="Test", role_description="Test")
            
            # Check if callbacks were passed to constructor
            _, kwargs = MockChat.call_args
            assert 'callbacks' in kwargs
            callbacks = kwargs['callbacks']
            assert len(callbacks) > 0
            
            # Verify it's our CostTrackingCallback
            from app.core.cost_tracker import CostTrackingCallback
            assert any(isinstance(c, CostTrackingCallback) for c in callbacks)

    def test_telemetry_integration(self):
        # Verify TelemetryService is reachable
        telemetry = TelemetryService()
        initial_count = telemetry.get_metrics()["llm_calls_total"]
        
        telemetry.track_llm_call("test-model", 10, 0.001)
        
        new_count = telemetry.get_metrics()["llm_calls_total"]
        assert new_count == initial_count + 1
