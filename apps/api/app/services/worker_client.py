"""
Worker Client Service
Triggers background processing tasks via Celery
"""
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Redis connection URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery client (same config as worker)
celery_app = Celery(
    "nerdlearn_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)


def trigger_module_processing(
    module_id: int, course_id: int, module_type: str, file_url: str, title: str
) -> str:
    """
    Trigger background processing for a module

    Args:
        module_id: Module ID
        course_id: Course ID
        module_type: Type of module (pdf, video)
        file_url: File URL in MinIO
        title: Module title

    Returns:
        Celery task ID
    """
    # Extract file path from URL (MinIO path)
    # URL format: http://minio:9000/nerdlearn/courses/1/modules/file.pdf
    # We need: courses/1/modules/file.pdf
    file_path = file_url.split(f"/{os.getenv('MINIO_BUCKET', 'nerdlearn')}/")[-1]

    # Trigger appropriate task based on module type
    if module_type == "pdf":
        task = celery_app.send_task(
            "process_pdf",
            args=[module_id, course_id, file_path, title],
            queue="documents",
        )
    elif module_type == "video":
        task = celery_app.send_task(
            "process_video",
            args=[module_id, course_id, file_path, title],
            queue="videos",
        )
    else:
        raise ValueError(f"Unsupported module type: {module_type}")

    return task.id


def get_task_status(task_id: str) -> dict:
    """
    Get the status of a background task

    Args:
        task_id: Celery task ID

    Returns:
        Task status information
    """
    result = celery_app.AsyncResult(task_id)

    return {
        "task_id": task_id,
        "status": result.state,
        "result": result.result if result.ready() else None,
        "info": result.info,
    }


def trigger_course_graph_build(course_id: int, course_title: str) -> str:
    """
    Trigger knowledge graph construction for a course

    Args:
        course_id: Course ID
        course_title: Course title

    Returns:
        Celery task ID
    """
    task = celery_app.send_task(
        "build_course_graph",
        args=[course_id, course_title],
        queue="processing",
    )

    return task.id
