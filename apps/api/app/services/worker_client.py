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


def get_task_status(task_id: str) -> str:
    """
    Get the status of a background task.

    Args:
        task_id: Celery task ID

    Returns:
        Task state string (PENDING, STARTED, PROCESSING, SUCCESS, FAILURE, etc.)
    """
    result = celery_app.AsyncResult(task_id)
    return result.state


def get_task_progress(task_id: str) -> dict:
    """
    Get detailed progress information for a task.

    Args:
        task_id: Celery task ID

    Returns:
        Progress information including step, percentage, retry count, etc.
    """
    result = celery_app.AsyncResult(task_id)

    progress = {
        "task_id": task_id,
        "status": result.state,
        "ready": result.ready(),
    }

    # Add progress info from task meta
    if result.info:
        if isinstance(result.info, dict):
            progress.update({
                "step": result.info.get("step"),
                "retry_count": result.info.get("retry_count", 0),
                "error": result.info.get("error"),
                "progress_percentage": result.info.get("progress"),
                "current_module": result.info.get("current_module"),
                "module_count": result.info.get("module_count"),
            })
        elif isinstance(result.info, Exception):
            progress["error"] = str(result.info)

    # Add result if completed
    if result.ready():
        if result.successful():
            progress["result"] = result.result
        else:
            progress["error"] = str(result.result) if result.result else "Task failed"

    return progress


def cancel_task(task_id: str) -> bool:
    """
    Cancel a running task.

    Args:
        task_id: Celery task ID

    Returns:
        True if cancel signal was sent successfully
    """
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return True
    except Exception:
        return False


def get_task_result(task_id: str, timeout: float = None) -> dict:
    """
    Get the result of a completed task.

    Args:
        task_id: Celery task ID
        timeout: Optional timeout to wait for result

    Returns:
        Task result or status if not ready
    """
    result = celery_app.AsyncResult(task_id)

    if timeout:
        try:
            return {
                "status": "SUCCESS",
                "result": result.get(timeout=timeout),
            }
        except Exception as e:
            return {
                "status": result.state,
                "error": str(e),
            }

    if result.ready():
        if result.successful():
            return {
                "status": "SUCCESS",
                "result": result.result,
            }
        else:
            return {
                "status": "FAILURE",
                "error": str(result.result),
            }

    return {
        "status": result.state,
        "ready": False,
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
