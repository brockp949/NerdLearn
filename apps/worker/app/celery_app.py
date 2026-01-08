"""
Celery application configuration
"""
from celery import Celery
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
    ],
)

# Celery configuration
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
)

# Task routing
app.conf.task_routes = {
    "app.tasks.pdf_tasks.*": {"queue": "documents"},
    "app.tasks.video_tasks.*": {"queue": "videos"},
    "app.tasks.chunking_tasks.*": {"queue": "processing"},
    "app.tasks.graph_tasks.*": {"queue": "processing"},
}

if __name__ == "__main__":
    app.start()
