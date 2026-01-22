"""
Webhook Callbacks Service

Provides webhook notification system for:
- Processing status updates
- Course completion events
- Achievement unlocks
- Assessment results
- System alerts

Features:
- Configurable webhook endpoints
- Event filtering
- Retry with exponential backoff
- Signature verification
- Event history logging
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import aiohttp

logger = logging.getLogger(__name__)


# ==================== Enums and Data Classes ====================

class WebhookEventType(str, Enum):
    """Types of webhook events"""
    # Processing events
    PROCESSING_STARTED = "processing.started"
    PROCESSING_PROGRESS = "processing.progress"
    PROCESSING_COMPLETED = "processing.completed"
    PROCESSING_FAILED = "processing.failed"

    # Course events
    COURSE_ENROLLED = "course.enrolled"
    COURSE_STARTED = "course.started"
    COURSE_COMPLETED = "course.completed"
    MODULE_COMPLETED = "module.completed"

    # Learning events
    CONCEPT_MASTERED = "concept.mastered"
    REVIEW_DUE = "review.due"
    STREAK_ACHIEVED = "streak.achieved"
    STREAK_LOST = "streak.lost"

    # Achievement events
    ACHIEVEMENT_UNLOCKED = "achievement.unlocked"
    BADGE_EARNED = "badge.earned"
    LEVEL_UP = "level.up"

    # Assessment events
    QUIZ_COMPLETED = "quiz.completed"
    ASSESSMENT_SUBMITTED = "assessment.submitted"
    GRADE_AVAILABLE = "grade.available"

    # Social events
    FRIEND_REQUEST = "friend.request"
    CHALLENGE_RECEIVED = "challenge.received"
    CHALLENGE_COMPLETED = "challenge.completed"

    # System events
    SYSTEM_MAINTENANCE = "system.maintenance"
    API_LIMIT_REACHED = "api.limit_reached"
    ERROR_OCCURRED = "error.occurred"


class WebhookStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    id: str
    url: str
    secret: str  # For signature verification
    events: List[WebhookEventType]  # Events to receive
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Delivery settings
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30


@dataclass
class WebhookEvent:
    """A webhook event to be delivered"""
    id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "nerdlearn"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "version": self.version,
            "data": self.payload
        }


@dataclass
class WebhookDelivery:
    """Record of webhook delivery attempt"""
    id: str
    endpoint_id: str
    event_id: str
    status: WebhookStatus
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None


# ==================== Webhook Service ====================

class WebhookService:
    """
    Manages webhook subscriptions and event delivery.
    """

    MAX_HISTORY_PER_ENDPOINT = 100
    MAX_RETRY_ATTEMPTS = 5
    BASE_RETRY_DELAY = 60  # seconds

    def __init__(self):
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._event_history: Dict[str, deque] = {}  # endpoint_id -> recent events
        self._retry_queue: List[WebhookDelivery] = []
        self._event_counter = 0

    def register_endpoint(self, endpoint: WebhookEndpoint) -> str:
        """
        Register a webhook endpoint.

        Args:
            endpoint: Webhook endpoint configuration

        Returns:
            Endpoint ID
        """
        self._endpoints[endpoint.id] = endpoint
        self._event_history[endpoint.id] = deque(maxlen=self.MAX_HISTORY_PER_ENDPOINT)
        logger.info(f"Registered webhook endpoint: {endpoint.id} -> {endpoint.url}")
        return endpoint.id

    def update_endpoint(
        self,
        endpoint_id: str,
        updates: Dict[str, Any]
    ) -> Optional[WebhookEndpoint]:
        """Update endpoint configuration"""
        endpoint = self._endpoints.get(endpoint_id)
        if not endpoint:
            return None

        for key, value in updates.items():
            if hasattr(endpoint, key):
                if key == "events":
                    value = [WebhookEventType(e) if isinstance(e, str) else e for e in value]
                setattr(endpoint, key, value)

        return endpoint

    def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete a webhook endpoint"""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
            self._event_history.pop(endpoint_id, None)
            logger.info(f"Deleted webhook endpoint: {endpoint_id}")
            return True
        return False

    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get endpoint by ID"""
        return self._endpoints.get(endpoint_id)

    def list_endpoints(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List all endpoints"""
        endpoints = self._endpoints.values()
        if active_only:
            endpoints = [e for e in endpoints if e.active]

        return [
            {
                "id": e.id,
                "url": e.url,
                "events": [ev.value for ev in e.events],
                "active": e.active,
                "created_at": e.created_at.isoformat()
            }
            for e in endpoints
        ]

    async def emit_event(
        self,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        target_endpoints: Optional[List[str]] = None
    ) -> List[str]:
        """
        Emit a webhook event.

        Args:
            event_type: Type of event
            payload: Event payload data
            target_endpoints: Optional list of specific endpoint IDs

        Returns:
            List of delivery IDs
        """
        self._event_counter += 1
        event = WebhookEvent(
            id=f"evt_{self._event_counter}_{int(time.time())}",
            event_type=event_type,
            payload=payload
        )

        # Find matching endpoints
        if target_endpoints:
            endpoints = [
                self._endpoints[eid]
                for eid in target_endpoints
                if eid in self._endpoints
            ]
        else:
            endpoints = [
                e for e in self._endpoints.values()
                if e.active and event_type in e.events
            ]

        # Deliver to each endpoint
        delivery_ids = []
        for endpoint in endpoints:
            delivery_id = await self._deliver_event(event, endpoint)
            delivery_ids.append(delivery_id)

        return delivery_ids

    async def _deliver_event(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint
    ) -> str:
        """Deliver event to endpoint"""
        delivery_id = f"dlv_{event.id}_{endpoint.id}"

        delivery = WebhookDelivery(
            id=delivery_id,
            endpoint_id=endpoint.id,
            event_id=event.id,
            status=WebhookStatus.PENDING
        )
        self._deliveries[delivery_id] = delivery

        # Record in history
        self._event_history[endpoint.id].append({
            "event_id": event.id,
            "type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "delivery_id": delivery_id
        })

        # Attempt delivery
        success = await self._attempt_delivery(event, endpoint, delivery)

        if not success and delivery.attempts < endpoint.max_retries:
            # Schedule retry
            self._schedule_retry(delivery, endpoint)

        return delivery_id

    async def _attempt_delivery(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery
    ) -> bool:
        """Attempt to deliver webhook"""
        delivery.attempts += 1
        delivery.last_attempt = datetime.utcnow()

        payload = event.to_dict()
        payload_json = json.dumps(payload, default=str)

        # Generate signature
        signature = self._generate_signature(payload_json, endpoint.secret)

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Event": event.event_type.value,
            "X-Webhook-Delivery": delivery.id,
            "User-Agent": "NerdLearn-Webhooks/1.0"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint.url,
                    data=payload_json,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
                ) as response:
                    delivery.response_code = response.status
                    delivery.response_body = await response.text()

                    if 200 <= response.status < 300:
                        delivery.status = WebhookStatus.DELIVERED
                        delivery.delivered_at = datetime.utcnow()
                        logger.info(f"Webhook delivered: {delivery.id}")
                        return True
                    else:
                        delivery.status = WebhookStatus.FAILED
                        delivery.error_message = f"HTTP {response.status}"
                        logger.warning(f"Webhook failed: {delivery.id} - {response.status}")
                        return False

        except asyncio.TimeoutError:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Request timed out"
            logger.warning(f"Webhook timeout: {delivery.id}")
            return False

        except Exception as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            logger.error(f"Webhook error: {delivery.id} - {e}")
            return False

    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """Verify webhook signature"""
        expected = self._generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected)

    def _schedule_retry(
        self,
        delivery: WebhookDelivery,
        endpoint: WebhookEndpoint
    ):
        """Schedule delivery retry with exponential backoff"""
        delivery.status = WebhookStatus.RETRYING

        # Exponential backoff
        delay = self.BASE_RETRY_DELAY * (2 ** (delivery.attempts - 1))
        delay = min(delay, 3600)  # Max 1 hour

        delivery.next_retry = datetime.utcnow() + timedelta(seconds=delay)
        self._retry_queue.append(delivery)

        logger.info(f"Scheduled retry for {delivery.id} in {delay}s")

    async def process_retry_queue(self):
        """Process pending retries"""
        now = datetime.utcnow()
        to_retry = [
            d for d in self._retry_queue
            if d.next_retry and d.next_retry <= now
        ]

        for delivery in to_retry:
            self._retry_queue.remove(delivery)

            endpoint = self._endpoints.get(delivery.endpoint_id)
            if not endpoint:
                continue

            # Reconstruct event (simplified - in production, store event data)
            event = WebhookEvent(
                id=delivery.event_id,
                event_type=WebhookEventType.PROCESSING_COMPLETED,
                payload={"retry": True}
            )

            success = await self._attempt_delivery(event, endpoint, delivery)

            if not success and delivery.attempts < self.MAX_RETRY_ATTEMPTS:
                self._schedule_retry(delivery, endpoint)
            elif not success:
                delivery.status = WebhookStatus.FAILED
                logger.error(f"Webhook permanently failed after {delivery.attempts} attempts: {delivery.id}")

    def get_delivery_status(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        """Get delivery status"""
        delivery = self._deliveries.get(delivery_id)
        if not delivery:
            return None

        return {
            "id": delivery.id,
            "endpoint_id": delivery.endpoint_id,
            "event_id": delivery.event_id,
            "status": delivery.status.value,
            "attempts": delivery.attempts,
            "last_attempt": delivery.last_attempt.isoformat() if delivery.last_attempt else None,
            "next_retry": delivery.next_retry.isoformat() if delivery.next_retry else None,
            "response_code": delivery.response_code,
            "error_message": delivery.error_message,
            "delivered_at": delivery.delivered_at.isoformat() if delivery.delivered_at else None
        }

    def get_event_history(
        self,
        endpoint_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent event history for endpoint"""
        history = self._event_history.get(endpoint_id, [])
        return list(history)[-limit:]

    def get_delivery_stats(
        self,
        endpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get delivery statistics"""
        if endpoint_id:
            deliveries = [
                d for d in self._deliveries.values()
                if d.endpoint_id == endpoint_id
            ]
        else:
            deliveries = list(self._deliveries.values())

        total = len(deliveries)
        delivered = sum(1 for d in deliveries if d.status == WebhookStatus.DELIVERED)
        failed = sum(1 for d in deliveries if d.status == WebhookStatus.FAILED)
        pending = sum(1 for d in deliveries if d.status in [WebhookStatus.PENDING, WebhookStatus.RETRYING])

        return {
            "total": total,
            "delivered": delivered,
            "failed": failed,
            "pending": pending,
            "success_rate": delivered / total if total > 0 else 0,
            "avg_attempts": sum(d.attempts for d in deliveries) / total if total > 0 else 0
        }


# ==================== Convenience Functions ====================

# Singleton instance
webhook_service = WebhookService()


async def emit_processing_event(
    content_id: str,
    status: str,
    progress: Optional[float] = None,
    error: Optional[str] = None
):
    """Emit processing status event"""
    if status == "started":
        event_type = WebhookEventType.PROCESSING_STARTED
    elif status == "completed":
        event_type = WebhookEventType.PROCESSING_COMPLETED
    elif status == "failed":
        event_type = WebhookEventType.PROCESSING_FAILED
    else:
        event_type = WebhookEventType.PROCESSING_PROGRESS

    payload = {
        "content_id": content_id,
        "status": status,
        "progress": progress,
        "error": error,
        "timestamp": datetime.utcnow().isoformat()
    }

    await webhook_service.emit_event(event_type, payload)


async def emit_course_event(
    user_id: str,
    course_id: str,
    event_type: str,
    details: Optional[Dict[str, Any]] = None
):
    """Emit course-related event"""
    type_map = {
        "enrolled": WebhookEventType.COURSE_ENROLLED,
        "started": WebhookEventType.COURSE_STARTED,
        "completed": WebhookEventType.COURSE_COMPLETED,
        "module_completed": WebhookEventType.MODULE_COMPLETED,
    }

    webhook_type = type_map.get(event_type)
    if not webhook_type:
        return

    payload = {
        "user_id": user_id,
        "course_id": course_id,
        "event": event_type,
        **(details or {}),
        "timestamp": datetime.utcnow().isoformat()
    }

    await webhook_service.emit_event(webhook_type, payload)


async def emit_achievement_event(
    user_id: str,
    achievement_type: str,
    achievement_id: str,
    details: Optional[Dict[str, Any]] = None
):
    """Emit achievement event"""
    type_map = {
        "achievement": WebhookEventType.ACHIEVEMENT_UNLOCKED,
        "badge": WebhookEventType.BADGE_EARNED,
        "level_up": WebhookEventType.LEVEL_UP,
    }

    webhook_type = type_map.get(achievement_type, WebhookEventType.ACHIEVEMENT_UNLOCKED)

    payload = {
        "user_id": user_id,
        "achievement_type": achievement_type,
        "achievement_id": achievement_id,
        **(details or {}),
        "timestamp": datetime.utcnow().isoformat()
    }

    await webhook_service.emit_event(webhook_type, payload)
