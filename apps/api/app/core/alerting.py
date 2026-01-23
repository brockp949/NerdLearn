from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from app.core.telemetry import TelemetryService

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Monitors system metrics and triggers alerts based on thresholds.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlertManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.telemetry = TelemetryService()
        self._alerts: List[Dict[str, Any]] = []
        self._initialized = True
        
        # Default thresholds
        self.thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "cost_spike": 5.00   # > $5.00 in single session (simulated)
        }

    def check_system_health(self) -> List[Dict[str, Any]]:
        """
        Analyze current telemetry and generate new alerts if needed.
        Returns: List of *newly* triggered alerts.
        """
        metrics = self.telemetry.get_metrics()
        new_alerts = []

        # 1. Check Error Rate
        total_calls = metrics.get("llm_calls_total", 0)
        total_errors = metrics.get("errors_total", 0)
        
        if total_calls > 10: # Minimum sample size
            error_rate = total_errors / total_calls
            if error_rate > self.thresholds["error_rate"]:
                alert = self._create_alert(
                    "High Error Rate", 
                    "critical", 
                    f"LLM Error rate is {error_rate:.1%} (Threshold: {self.thresholds['error_rate']:.1%})"
                )
                if alert:
                    new_alerts.append(alert)

        # 2. Check Cost Spike (Naive check against total for now)
        # In a real system, this would check rate of change
        current_cost = metrics.get("llm_cost_est_usd", 0.0)
        if current_cost > self.thresholds["cost_spike"]:
             # Simple dedup: don't alert if we already have a cost alert for this session
             # Real implementation would be time-windowed
             if not any(a["type"] == "High Cost" for a in self._alerts):
                alert = self._create_alert(
                    "High Cost",
                    "warning",
                    f"Total session cost exceeded ${self.thresholds['cost_spike']:.2f}"
                )
                if alert:
                    new_alerts.append(alert)
                    
        return new_alerts

    def _create_alert(self, title: str, severity: str, message: str) -> Optional[Dict[str, Any]]:
        """Internal helper to create and log an alert."""
        
        # Basic de-duplication: don't trigger identical alert if it's the most recent one
        if self._alerts and self._alerts[-1]["type"] == title and \
           (datetime.utcnow() - datetime.fromisoformat(self._alerts[-1]["timestamp"])).total_seconds() < 300:
            return None

        alert = {
            "id": len(self._alerts) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "type": title,
            "severity": severity,
            "message": message,
            "status": "active"
        }
        
        self._alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}]: {title} - {message}")
        return alert

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        return [a for a in self._alerts if a["status"] == "active"]

    def resolve_alert(self, alert_id: int):
        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["status"] = "resolved"
                alert["resolved_at"] = datetime.utcnow().isoformat()
