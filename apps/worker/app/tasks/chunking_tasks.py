"""
Chunking and Embedding Tasks with Exponential Backoff Retry
"""
import logging
from ..celery_app import app, DEFAULT_RETRY_KWARGS, RETRYABLE_EXCEPTIONS
from ..services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="reindex_module",
    **DEFAULT_RETRY_KWARGS,
)
def reindex_module(self, module_id: int, course_id: int):
    """
    Reindex a module (regenerate embeddings) with automatic retry on transient failures.

    Retry behavior:
    - Retries on: ConnectionError, TimeoutError, OSError, IOError
    - Exponential backoff: 2s, 4s, 8s, 16s, 32s (capped at 600s)
    - Jitter added to prevent thundering herd
    - Max 5 retries before permanent failure

    Args:
        module_id: Module ID
        course_id: Course ID
    """
    retry_count = self.request.retries
    if retry_count > 0:
        logger.info(f"Retry attempt {retry_count}/{self.max_retries} for reindex module {module_id}")

    try:
        self.update_state(state="PROCESSING", meta={"step": "Deleting old embeddings", "retry_count": retry_count})

        vector_store = VectorStoreService()

        # Delete existing chunks
        vector_store.delete_module_chunks(module_id)

        # Note: This would need to re-trigger processing based on module type
        # For now, we just delete. Full re-indexing would call process_pdf or process_video

        return {
            "status": "success",
            "message": "Module embeddings deleted. Re-upload module to reindex.",
            "retries_used": retry_count,
        }

    except RETRYABLE_EXCEPTIONS as e:
        logger.warning(f"Retryable error reindexing module {module_id}: {e}")
        self.update_state(
            state="RETRYING",
            meta={"error": str(e), "retry_count": retry_count + 1}
        )
        raise

    except Exception as e:
        logger.error(f"Non-retryable error reindexing module {module_id}: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e), "retry_count": retry_count})
        raise
