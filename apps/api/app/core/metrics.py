"""
Prometheus Metrics for NerdLearn API

Provides observability metrics for monitoring and alerting.
"""
from typing import Callable
import time
import logging

logger = logging.getLogger(__name__)

# Metrics storage (simplified - in production use prometheus_client)
_metrics = {
    "http_requests_total": {},
    "http_request_duration_seconds": [],
    "active_users": 0,
    "db_connections": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}


def increment_counter(name: str, labels: dict = None):
    """Increment a counter metric"""
    key = f"{name}:{labels}" if labels else name
    if name not in _metrics:
        _metrics[name] = {}
    if isinstance(_metrics[name], dict):
        _metrics[name][key] = _metrics[name].get(key, 0) + 1


def observe_histogram(name: str, value: float, labels: dict = None):
    """Record a histogram observation"""
    if name not in _metrics:
        _metrics[name] = []
    _metrics[name].append({"value": value, "labels": labels, "time": time.time()})


def set_gauge(name: str, value: float):
    """Set a gauge metric"""
    _metrics[name] = value


def get_metrics() -> dict:
    """Get all metrics for export"""
    return _metrics.copy()


class MetricsMiddleware:
    """
    FastAPI middleware for automatic request metrics.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")

        # Process request
        status_code = 500
        try:
            # Capture status code from response
            async def send_wrapper(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            await self.app(scope, receive, send_wrapper)
        finally:
            # Record metrics
            duration = time.time() - start_time
            labels = {"method": method, "endpoint": path, "status_code": str(status_code)}

            increment_counter("http_requests_total", labels)
            observe_histogram("http_request_duration_seconds", duration, labels)


# Metrics endpoint helper
def format_prometheus_metrics() -> str:
    """Format metrics in Prometheus text format"""
    lines = []

    # Request counter
    lines.append("# HELP http_requests_total Total HTTP requests")
    lines.append("# TYPE http_requests_total counter")
    if isinstance(_metrics.get("http_requests_total"), dict):
        for key, value in _metrics["http_requests_total"].items():
            lines.append(f'http_requests_total{{{key}}} {value}')

    # Request duration histogram
    lines.append("# HELP http_request_duration_seconds HTTP request duration")
    lines.append("# TYPE http_request_duration_seconds histogram")
    durations = _metrics.get("http_request_duration_seconds", [])
    if durations:
        total = sum(d["value"] for d in durations)
        count = len(durations)
        lines.append(f"http_request_duration_seconds_sum {total}")
        lines.append(f"http_request_duration_seconds_count {count}")

    # Gauges
    for gauge_name in ["active_users", "db_connections"]:
        value = _metrics.get(gauge_name, 0)
        lines.append(f"# HELP {gauge_name} Current {gauge_name.replace('_', ' ')}")
        lines.append(f"# TYPE {gauge_name} gauge")
        lines.append(f"{gauge_name} {value}")

    return "\n".join(lines)
