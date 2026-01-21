"""
Celery application configuration with retry support and beat scheduling
"""
from celery import Celery
from celery.schedules import crontab
import os
from dotenv import load_dotenv

load_dotenv()

# Redis connection URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery(
    "nerdlearn_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.pdf_tasks",
        "app.tasks.video_tasks",
        "app.tasks.chunking_tasks",
        "app.tasks.graph_tasks",
        "app.tasks.batch_tasks",
        "app.tasks.fsrs_tasks",
    ],
)

# Celery configuration with retry settings
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (memory management)
    # Retry configuration
    task_acks_late=True,  # Acknowledge after task completes (for reliability)
    task_reject_on_worker_lost=True,  # Reject task if worker dies
    task_default_retry_delay=2,  # Initial retry delay (seconds)
)


# Retryable exceptions for automatic retry
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
    IOError,
)

# Default retry settings for tasks
DEFAULT_RETRY_KWARGS = {
    "autoretry_for": RETRYABLE_EXCEPTIONS,
    "retry_backoff": True,  # Exponential backoff
    "retry_backoff_max": 600,  # Max 10 minutes between retries
    "retry_jitter": True,  # Add randomness to prevent thundering herd
    "max_retries": 5,  # Maximum 5 retries
}

# Task routing
app.conf.task_routes = {
    "app.tasks.pdf_tasks.*": {"queue": "documents"},
    "app.tasks.video_tasks.*": {"queue": "videos"},
    "app.tasks.chunking_tasks.*": {"queue": "processing"},
    "app.tasks.graph_tasks.*": {"queue": "processing"},
    "app.tasks.fsrs_tasks.*": {"queue": "optimization"},
}

# Celery beat schedule for periodic tasks
app.conf.beat_schedule = {
    # FSRS parameter optimization - runs weekly on Sunday at 3 AM UTC
    "weekly-fsrs-optimization": {
        "task": "scheduled_fsrs_optimization",
        "schedule": crontab(hour=3, minute=0, day_of_week=0),
        "options": {"queue": "optimization"},
    },
    # Daily maintenance tasks
    "daily-cleanup-expired-sessions": {
        "task": "cleanup_expired_sessions",
        "schedule": crontab(hour=4, minute=0),
        "options": {"queue": "maintenance"},
    },
}

if __name__ == "__main__":
    app.start()
