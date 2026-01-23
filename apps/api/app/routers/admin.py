from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from datetime import datetime

from app.core.telemetry import TelemetryService
from app.core.alerting import AlertManager

router = APIRouter()

@router.get("/health")
async def get_health_status() -> Dict[str, Any]:
    """
    Get system health status.
    """
    telemetry = TelemetryService()
    metrics = telemetry.get_metrics()
    
    # Basic check - if we can get metrics, the service is up
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_start": metrics.get("start_time"),
        "services": {
            "api": "up",
            "telemetry": "up",
            # DB check could go here
        }
    }

@router.get("/metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """
    Get raw system metrics from TelemetryService.
    """
    telemetry = TelemetryService()
    return {
        "metrics": telemetry.get_metrics(),
        "recent_actions": telemetry.get_recent_actions(limit=20)
    }

@router.get("/alerts")
async def get_system_alerts() -> List[Dict[str, Any]]:
    """
    Get active system alerts.
    Also triggers a health check to look for new alerts.
    """
    manager = AlertManager()
    
    # Trigger a check just case
    manager.check_system_health()
    
    return manager.get_active_alerts()

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int) -> Dict[str, str]:
    """
    Resolve a specific alert.
    """
    manager = AlertManager()
    manager.resolve_alert(alert_id)
    return {"status": "resolved"}
