"""
Distributed Telemetry Collection System

Scalable telemetry collection for learning analytics:
- Multi-node event collection
- Event batching and aggregation
- Real-time streaming
- Offline buffering
- Privacy-preserving analytics

Architecture:
- Collectors: Edge nodes that gather events
- Aggregators: Process and summarize data
- Storage: Time-series optimized storage
- Exporters: Push to external systems
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import gzip

logger = logging.getLogger(__name__)


# ==================== Enums and Data Classes ====================

class TelemetryLevel(str, Enum):
    """Telemetry detail levels"""
    MINIMAL = "minimal"      # Basic counts only
    STANDARD = "standard"    # Standard analytics
    DETAILED = "detailed"    # Full event data
    DEBUG = "debug"          # Everything including debug info


class EventPriority(str, Enum):
    """Event priority for processing"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AggregationType(str, Enum):
    """Types of aggregation"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"
    UNIQUE_COUNT = "unique_count"


@dataclass
class TelemetryEvent:
    """A single telemetry event"""
    event_id: str
    event_type: str
    timestamp: datetime
    user_id: Optional[str] = None  # Anonymized
    session_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    priority: EventPriority = EventPriority.NORMAL
    node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "properties": self.properties,
            "metrics": self.metrics,
            "tags": self.tags,
            "priority": self.priority.value,
            "node_id": self.node_id
        }

    def to_bytes(self) -> bytes:
        """Serialize for transmission"""
        return json.dumps(self.to_dict()).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> "TelemetryEvent":
        """Deserialize from bytes"""
        d = json.loads(data.decode('utf-8'))
        return cls(
            event_id=d["event_id"],
            event_type=d["event_type"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            user_id=d.get("user_id"),
            session_id=d.get("session_id"),
            properties=d.get("properties", {}),
            metrics=d.get("metrics", {}),
            tags=d.get("tags", []),
            priority=EventPriority(d.get("priority", "normal")),
            node_id=d.get("node_id")
        )


@dataclass
class AggregatedMetric:
    """Aggregated metric result"""
    name: str
    aggregation_type: AggregationType
    value: float
    count: int
    period_start: datetime
    period_end: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryBatch:
    """Batch of telemetry events"""
    batch_id: str
    events: List[TelemetryEvent]
    created_at: datetime
    node_id: str
    compressed: bool = False

    def compress(self) -> bytes:
        """Compress batch for transmission"""
        data = json.dumps([e.to_dict() for e in self.events]).encode('utf-8')
        return gzip.compress(data)

    @classmethod
    def decompress(cls, data: bytes, batch_id: str, node_id: str) -> "TelemetryBatch":
        """Decompress batch"""
        json_data = gzip.decompress(data)
        events_data = json.loads(json_data)
        events = [
            TelemetryEvent(
                event_id=e["event_id"],
                event_type=e["event_type"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                user_id=e.get("user_id"),
                session_id=e.get("session_id"),
                properties=e.get("properties", {}),
                metrics=e.get("metrics", {}),
                tags=e.get("tags", []),
                priority=EventPriority(e.get("priority", "normal")),
                node_id=e.get("node_id")
            )
            for e in events_data
        ]
        return cls(
            batch_id=batch_id,
            events=events,
            created_at=datetime.utcnow(),
            node_id=node_id,
            compressed=True
        )


# ==================== Storage Backend ====================

class TelemetryStorageBackend(ABC):
    """Abstract base for telemetry storage"""

    @abstractmethod
    async def store_events(self, events: List[TelemetryEvent]):
        """Store events"""
        pass

    @abstractmethod
    async def store_aggregation(self, metric: AggregatedMetric):
        """Store aggregated metric"""
        pass

    @abstractmethod
    async def query_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[TelemetryEvent]:
        """Query events"""
        pass

    @abstractmethod
    async def query_aggregations(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        """Query aggregations"""
        pass


class InMemoryStorage(TelemetryStorageBackend):
    """In-memory storage for development/testing"""

    def __init__(self, max_events: int = 100000):
        self._events: deque = deque(maxlen=max_events)
        self._aggregations: Dict[str, List[AggregatedMetric]] = defaultdict(list)

    async def store_events(self, events: List[TelemetryEvent]):
        self._events.extend(events)

    async def store_aggregation(self, metric: AggregatedMetric):
        self._aggregations[metric.name].append(metric)

    async def query_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[TelemetryEvent]:
        results = []
        for event in self._events:
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return results

    async def query_aggregations(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        results = []
        for metric in self._aggregations.get(metric_name, []):
            if metric.period_start < start_time or metric.period_end > end_time:
                continue
            if dimensions:
                if not all(
                    metric.dimensions.get(k) == v
                    for k, v in dimensions.items()
                ):
                    continue
            results.append(metric)
        return results


# ==================== Collector ====================

class TelemetryCollector:
    """
    Collects telemetry events from applications.

    Features:
    - Event batching
    - Offline buffering
    - Priority queuing
    - Automatic retry
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_buffer_size: int = 10000,
        level: TelemetryLevel = TelemetryLevel.STANDARD
    ):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.level = level

        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._priority_queue: Dict[EventPriority, deque] = {
            p: deque() for p in EventPriority
        }
        self._handlers: List[Callable[[TelemetryBatch], None]] = []
        self._running = False
        self._event_counter = 0
        self._dropped_events = 0
        self._flush_task: Optional[asyncio.Task] = None

    def add_handler(self, handler: Callable[[TelemetryBatch], None]):
        """Add batch handler"""
        self._handlers.append(handler)

    def track(
        self,
        event_type: str,
        properties: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        priority: EventPriority = EventPriority.NORMAL
    ):
        """
        Track a telemetry event.

        Args:
            event_type: Type of event
            properties: Event properties
            metrics: Numeric metrics
            user_id: Anonymized user ID
            session_id: Session ID
            tags: Event tags
            priority: Event priority
        """
        # Check level
        if self.level == TelemetryLevel.MINIMAL:
            properties = {}
            metrics = {k: v for k, v in (metrics or {}).items() if k in ["count", "duration"]}

        self._event_counter += 1
        event = TelemetryEvent(
            event_id=f"{self.node_id}_{self._event_counter}",
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=self._anonymize_user_id(user_id) if user_id else None,
            session_id=session_id,
            properties=properties or {},
            metrics=metrics or {},
            tags=tags or [],
            priority=priority,
            node_id=self.node_id
        )

        # Add to appropriate queue
        if priority == EventPriority.CRITICAL:
            # Flush critical events immediately
            asyncio.create_task(self._flush_single(event))
        else:
            self._priority_queue[priority].append(event)

            # Check if should flush
            total_queued = sum(len(q) for q in self._priority_queue.values())
            if total_queued >= self.batch_size:
                asyncio.create_task(self._flush())

    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID for privacy"""
        # Use hash for anonymization
        return hashlib.sha256(f"nerdlearn_{user_id}".encode()).hexdigest()[:16]

    async def start(self):
        """Start the collector"""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"Telemetry collector started: {self.node_id}")

    async def stop(self):
        """Stop the collector"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush()
        logger.info(f"Telemetry collector stopped: {self.node_id}")

    async def _periodic_flush(self):
        """Periodically flush events"""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()

    async def _flush(self):
        """Flush queued events"""
        events = []

        # Collect from priority queues (high priority first)
        for priority in [EventPriority.HIGH, EventPriority.NORMAL, EventPriority.LOW]:
            while self._priority_queue[priority] and len(events) < self.batch_size:
                events.append(self._priority_queue[priority].popleft())

        if not events:
            return

        batch = TelemetryBatch(
            batch_id=f"batch_{self.node_id}_{int(time.time())}",
            events=events,
            created_at=datetime.utcnow(),
            node_id=self.node_id
        )

        # Send to handlers
        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(batch)
                else:
                    handler(batch)
            except Exception as e:
                logger.error(f"Handler error: {e}")
                # Buffer for retry
                self._buffer.extend(events)

    async def _flush_single(self, event: TelemetryEvent):
        """Flush a single critical event immediately"""
        batch = TelemetryBatch(
            batch_id=f"critical_{self.node_id}_{int(time.time())}",
            events=[event],
            created_at=datetime.utcnow(),
            node_id=self.node_id
        )

        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(batch)
                else:
                    handler(batch)
            except Exception as e:
                logger.error(f"Critical event handler error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return {
            "node_id": self.node_id,
            "events_tracked": self._event_counter,
            "events_dropped": self._dropped_events,
            "buffer_size": len(self._buffer),
            "queue_sizes": {
                p.value: len(q) for p, q in self._priority_queue.items()
            },
            "handlers": len(self._handlers),
            "running": self._running
        }


# ==================== Aggregator ====================

class TelemetryAggregator:
    """
    Aggregates telemetry events into metrics.

    Features:
    - Time-window aggregation
    - Multiple aggregation types
    - Dimension grouping
    - Real-time computation
    """

    def __init__(
        self,
        storage: TelemetryStorageBackend,
        window_size: timedelta = timedelta(minutes=1)
    ):
        self.storage = storage
        self.window_size = window_size

        # Aggregation buffers
        self._metric_buffers: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._window_start = datetime.utcnow()

    async def process_batch(self, batch: TelemetryBatch):
        """Process a batch of events"""
        # Store raw events
        await self.storage.store_events(batch.events)

        # Aggregate metrics
        for event in batch.events:
            for metric_name, value in event.metrics.items():
                # Create dimension key
                dim_key = json.dumps(event.properties, sort_keys=True)
                self._metric_buffers[metric_name][dim_key].append(value)

        # Check if window expired
        if datetime.utcnow() - self._window_start > self.window_size:
            await self._flush_aggregations()

    async def _flush_aggregations(self):
        """Flush aggregated metrics"""
        window_end = datetime.utcnow()

        for metric_name, dim_values in self._metric_buffers.items():
            for dim_key, values in dim_values.items():
                if not values:
                    continue

                dimensions = json.loads(dim_key) if dim_key != "{}" else {}

                # Compute aggregations
                aggregations = [
                    AggregatedMetric(
                        name=f"{metric_name}_count",
                        aggregation_type=AggregationType.COUNT,
                        value=len(values),
                        count=len(values),
                        period_start=self._window_start,
                        period_end=window_end,
                        dimensions=dimensions
                    ),
                    AggregatedMetric(
                        name=f"{metric_name}_sum",
                        aggregation_type=AggregationType.SUM,
                        value=sum(values),
                        count=len(values),
                        period_start=self._window_start,
                        period_end=window_end,
                        dimensions=dimensions
                    ),
                    AggregatedMetric(
                        name=f"{metric_name}_avg",
                        aggregation_type=AggregationType.AVERAGE,
                        value=sum(values) / len(values),
                        count=len(values),
                        period_start=self._window_start,
                        period_end=window_end,
                        dimensions=dimensions
                    ),
                ]

                if values:
                    aggregations.extend([
                        AggregatedMetric(
                            name=f"{metric_name}_min",
                            aggregation_type=AggregationType.MIN,
                            value=min(values),
                            count=len(values),
                            period_start=self._window_start,
                            period_end=window_end,
                            dimensions=dimensions
                        ),
                        AggregatedMetric(
                            name=f"{metric_name}_max",
                            aggregation_type=AggregationType.MAX,
                            value=max(values),
                            count=len(values),
                            period_start=self._window_start,
                            period_end=window_end,
                            dimensions=dimensions
                        ),
                    ])

                # Store aggregations
                for agg in aggregations:
                    await self.storage.store_aggregation(agg)

        # Reset buffers
        self._metric_buffers.clear()
        self._window_start = window_end

    async def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: AggregationType = AggregationType.AVERAGE,
        dimensions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Query aggregated metrics"""
        full_name = f"{metric_name}_{aggregation.value}"
        aggregations = await self.storage.query_aggregations(
            full_name, start_time, end_time, dimensions
        )

        return [
            {
                "name": agg.name,
                "value": agg.value,
                "count": agg.count,
                "period_start": agg.period_start.isoformat(),
                "period_end": agg.period_end.isoformat(),
                "dimensions": agg.dimensions
            }
            for agg in aggregations
        ]


# ==================== Distributed Coordinator ====================

class TelemetryCoordinator:
    """
    Coordinates distributed telemetry collection.

    Features:
    - Node registration
    - Load balancing
    - Health monitoring
    - Centralized querying
    """

    def __init__(self, storage: Optional[TelemetryStorageBackend] = None):
        self.storage = storage or InMemoryStorage()
        self.aggregator = TelemetryAggregator(self.storage)

        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._collectors: Dict[str, TelemetryCollector] = {}

    def register_node(
        self,
        node_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a collector node"""
        self._nodes[node_id] = {
            "node_id": node_id,
            "registered_at": datetime.utcnow(),
            "last_heartbeat": datetime.utcnow(),
            "status": "active",
            "metadata": metadata or {},
            "events_processed": 0
        }
        logger.info(f"Registered telemetry node: {node_id}")

    def heartbeat(self, node_id: str):
        """Update node heartbeat"""
        if node_id in self._nodes:
            self._nodes[node_id]["last_heartbeat"] = datetime.utcnow()

    def create_collector(
        self,
        node_id: Optional[str] = None,
        **kwargs
    ) -> TelemetryCollector:
        """Create a new collector"""
        collector = TelemetryCollector(node_id=node_id, **kwargs)

        # Add aggregator as handler
        async def handle_batch(batch: TelemetryBatch):
            await self.aggregator.process_batch(batch)
            if batch.node_id in self._nodes:
                self._nodes[batch.node_id]["events_processed"] += len(batch.events)

        collector.add_handler(handle_batch)

        # Register node
        self.register_node(collector.node_id)
        self._collectors[collector.node_id] = collector

        return collector

    def get_node_status(self) -> List[Dict[str, Any]]:
        """Get status of all nodes"""
        now = datetime.utcnow()
        return [
            {
                **node,
                "registered_at": node["registered_at"].isoformat(),
                "last_heartbeat": node["last_heartbeat"].isoformat(),
                "healthy": (now - node["last_heartbeat"]).total_seconds() < 60
            }
            for node in self._nodes.values()
        ]

    async def query_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query events across all nodes"""
        events = await self.storage.query_events(
            event_type, start_time, end_time, limit
        )
        return [e.to_dict() for e in events]

    async def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: AggregationType = AggregationType.AVERAGE
    ) -> List[Dict[str, Any]]:
        """Query metrics across all nodes"""
        return await self.aggregator.query_metrics(
            metric_name, start_time, end_time, aggregation
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "total_nodes": len(self._nodes),
            "active_nodes": sum(
                1 for n in self._nodes.values()
                if (datetime.utcnow() - n["last_heartbeat"]).total_seconds() < 60
            ),
            "total_events_processed": sum(
                n["events_processed"] for n in self._nodes.values()
            ),
            "collectors": {
                node_id: collector.get_stats()
                for node_id, collector in self._collectors.items()
            }
        }


# Singleton coordinator
telemetry_coordinator = TelemetryCoordinator()
