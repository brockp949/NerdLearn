"""
Batch Processing Tasks with Exponential Backoff Retry

Provides course-level batch processing for all modules.
"""
import logging
from celery import group, chord
from ..celery_app import app, DEFAULT_RETRY_KWARGS, RETRYABLE_EXCEPTIONS
from .pdf_tasks import process_pdf
from .video_tasks import process_video

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="process_course_batch",
    **DEFAULT_RETRY_KWARGS,
)
def process_course_batch(
    self,
    course_id: int,
    modules: list,
):
    """
    Process all modules in a course in parallel.

    Uses Celery group to run multiple module processing tasks concurrently.

    Args:
        course_id: Course ID
        modules: List of module dicts with keys:
            - id: Module ID
            - type: "pdf" or "video"
            - file_path: Path to file in MinIO
            - title: Module title

    Returns:
        Dict with processing results for each module
    """
    retry_count = self.request.retries
    if retry_count > 0:
        logger.info(f"Retry attempt {retry_count}/{self.max_retries} for course batch {course_id}")

    try:
        self.update_state(
            state="PROCESSING",
            meta={
                "step": "Preparing batch processing",
                "course_id": course_id,
                "module_count": len(modules),
                "retry_count": retry_count,
            }
        )

        if not modules:
            return {
                "status": "success",
                "course_id": course_id,
                "message": "No modules to process",
                "processed": 0,
                "results": [],
            }

        # Create processing tasks for each module
        tasks = []
        for module in modules:
            module_id = module["id"]
            module_type = module["type"].lower()
            file_path = module["file_path"]
            title = module["title"]

            if module_type == "pdf":
                task = process_pdf.s(module_id, course_id, file_path, title)
            elif module_type in ("video", "mp4", "webm", "mov"):
                task = process_video.s(module_id, course_id, file_path, title)
            else:
                logger.warning(f"Unknown module type {module_type} for module {module_id}")
                continue

            tasks.append(task)

        if not tasks:
            return {
                "status": "success",
                "course_id": course_id,
                "message": "No processable modules found",
                "processed": 0,
                "results": [],
            }

        self.update_state(
            state="PROCESSING",
            meta={
                "step": f"Processing {len(tasks)} modules in parallel",
                "course_id": course_id,
                "module_count": len(tasks),
                "retry_count": retry_count,
            }
        )

        # Execute all tasks in parallel using a group
        job = group(tasks)
        result = job.apply_async()

        # Wait for all tasks to complete
        results = result.get(timeout=3600)  # 1 hour timeout for entire batch

        # Aggregate results
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - successful

        return {
            "status": "success" if failed == 0 else "partial",
            "course_id": course_id,
            "processed": successful,
            "failed": failed,
            "total": len(results),
            "results": results,
            "retries_used": retry_count,
        }

    except RETRYABLE_EXCEPTIONS as e:
        logger.warning(f"Retryable error in course batch {course_id}: {e}")
        self.update_state(
            state="RETRYING",
            meta={"error": str(e), "retry_count": retry_count + 1}
        )
        raise

    except Exception as e:
        logger.error(f"Non-retryable error in course batch {course_id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "retry_count": retry_count}
        )
        raise


@app.task(
    bind=True,
    name="get_batch_status",
)
def get_batch_status(self, task_ids: list):
    """
    Get status of multiple processing tasks.

    Args:
        task_ids: List of Celery task IDs to check

    Returns:
        Dict with status of each task
    """
    from celery.result import AsyncResult

    statuses = []
    for task_id in task_ids:
        result = AsyncResult(task_id)
        statuses.append({
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "result": result.result if result.ready() and result.successful() else None,
            "error": str(result.result) if result.ready() and result.failed() else None,
        })

    completed = sum(1 for s in statuses if s["ready"])
    successful = sum(1 for s in statuses if s["successful"])

    return {
        "total": len(task_ids),
        "completed": completed,
        "successful": successful,
        "failed": completed - successful,
        "pending": len(task_ids) - completed,
        "progress_percentage": (completed / len(task_ids) * 100) if task_ids else 100,
        "tasks": statuses,
    }


@app.task(
    bind=True,
    name="process_course_sequential",
    **DEFAULT_RETRY_KWARGS,
)
def process_course_sequential(
    self,
    course_id: int,
    modules: list,
):
    """
    Process all modules in a course sequentially (one at a time).

    Useful for resource-constrained environments or when order matters.

    Args:
        course_id: Course ID
        modules: List of module dicts (same format as process_course_batch)

    Returns:
        Dict with processing results for each module
    """
    retry_count = self.request.retries
    results = []

    try:
        total_modules = len(modules)

        for i, module in enumerate(modules):
            module_id = module["id"]
            module_type = module["type"].lower()
            file_path = module["file_path"]
            title = module["title"]

            self.update_state(
                state="PROCESSING",
                meta={
                    "step": f"Processing module {i + 1}/{total_modules}: {title}",
                    "course_id": course_id,
                    "current_module": module_id,
                    "progress": i / total_modules * 100,
                    "retry_count": retry_count,
                }
            )

            try:
                if module_type == "pdf":
                    result = process_pdf(module_id, course_id, file_path, title)
                elif module_type in ("video", "mp4", "webm", "mov"):
                    result = process_video(module_id, course_id, file_path, title)
                else:
                    result = {"status": "skipped", "reason": f"Unknown type: {module_type}"}

                results.append({
                    "module_id": module_id,
                    "title": title,
                    **result,
                })
            except Exception as e:
                logger.error(f"Error processing module {module_id}: {e}")
                results.append({
                    "module_id": module_id,
                    "title": title,
                    "status": "failed",
                    "error": str(e),
                })

        successful = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "failed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")

        return {
            "status": "success" if failed == 0 else "partial",
            "course_id": course_id,
            "processed": successful,
            "failed": failed,
            "skipped": skipped,
            "total": len(results),
            "results": results,
            "retries_used": retry_count,
        }

    except RETRYABLE_EXCEPTIONS as e:
        logger.warning(f"Retryable error in sequential processing {course_id}: {e}")
        self.update_state(
            state="RETRYING",
            meta={"error": str(e), "retry_count": retry_count + 1, "partial_results": results}
        )
        raise

    except Exception as e:
        logger.error(f"Non-retryable error in sequential processing {course_id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "retry_count": retry_count, "partial_results": results}
        )
        raise
