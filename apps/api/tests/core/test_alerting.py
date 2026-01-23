import pytest
from app.core.alerting import AlertManager
from app.core.telemetry import TelemetryService
from unittest.mock import MagicMock

class TestAlerting:
    
    def setup_method(self):
        # Reset singletons
        TelemetryService().reset_metrics()
        # Reset alerts manually since we didn't implement reset_alerts on AlertManager
        AlertManager()._alerts = []
        
    def test_alert_thresholds(self):
        telemetry = TelemetryService()
        manager = AlertManager()
        
        # Simulate high error rate
        # 20 calls, 2 errors = 10% error rate (Threshold is 5%)
        for _ in range(18):
            telemetry.track_llm_call("test", 10, 0.01, success=True)
        for _ in range(2):
            telemetry.track_llm_call("test", 10, 0.01, success=False)
            
        new_alerts = manager.check_system_health()
        
        assert len(new_alerts) == 1
        assert new_alerts[0]["severity"] == "critical"
        assert "High Error Rate" in new_alerts[0]["type"]

    def test_alert_deduplication(self):
        telemetry = TelemetryService()
        manager = AlertManager()
        
        # Trigger alert
        for _ in range(20):
             telemetry.track_llm_call("test", 10, 0.01, success=False) # 100% error rate
             
        alerts1 = manager.check_system_health()
        assert len(alerts1) == 1
        
        # Check again immediately - should not create duplicate
        alerts2 = manager.check_system_health()
        assert len(alerts2) == 0
        
        # Active alerts should still be 1
        assert len(manager.get_active_alerts()) == 1
