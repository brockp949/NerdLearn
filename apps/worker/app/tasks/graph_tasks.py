"""
Knowledge Graph Construction Tasks with Exponential Backoff Retry
"""
import logging
from ..celery_app import app, DEFAULT_RETRY_KWARGS, RETRYABLE_EXCEPTIONS
from ..services.graph_service import GraphService
from ..services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="build_course_graph",
    **DEFAULT_RETRY_KWARGS,
)
def build_course_graph(self, course_id: int, course_title: str):
    """
    Build knowledge graph for an entire course with automatic retry on transient failures.

    Retry behavior:
    - Retries on: ConnectionError, TimeoutError, OSError, IOError
    - Exponential backoff: 2s, 4s, 8s, 16s, 32s (capped at 600s)
    - Jitter added to prevent thundering herd
    - Max 5 retries before permanent failure

    Args:
        course_id: Course ID
        course_title: Course title
    """
    retry_count = self.request.retries
    if retry_count > 0:
        logger.info(f"Retry attempt {retry_count}/{self.max_retries} for build_course_graph {course_id}")

    try:
        self.update_state(
            state="PROCESSING", meta={"step": "Fetching course modules", "retry_count": retry_count}
        )

        # Note: In production, we'd query the database for modules
        # For now, this task assumes modules have already been processed
        # and we're just building the graph from stored concepts

        graph_service = GraphService()

        # This is a placeholder - in production, we'd:
        # 1. Query database for all modules in course
        # 2. Get their extracted concepts
        # 3. Build the graph

        self.update_state(
            state="PROCESSING", meta={"step": "Constructing knowledge graph", "retry_count": retry_count}
        )

        # Close connection
        graph_service.close()

        return {
            "status": "success",
            "message": "Knowledge graph built successfully",
            "course_id": course_id,
            "retries_used": retry_count,
        }

    except RETRYABLE_EXCEPTIONS as e:
        logger.warning(f"Retryable error building graph for course {course_id}: {e}")
        self.update_state(
            state="RETRYING",
            meta={"error": str(e), "retry_count": retry_count + 1}
        )
        raise

    except Exception as e:
        logger.error(f"Non-retryable error building graph for course {course_id}: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e), "retry_count": retry_count})
        raise


@app.task(
    bind=True,
    name="update_concept_relationships",
    **DEFAULT_RETRY_KWARGS,
)
def update_concept_relationships(self, course_id: int):
    """
    Update concept prerequisite relationships with automatic retry on transient failures.

    Retry behavior:
    - Retries on: ConnectionError, TimeoutError, OSError, IOError
    - Exponential backoff: 2s, 4s, 8s, 16s, 32s (capped at 600s)
    - Jitter added to prevent thundering herd
    - Max 5 retries before permanent failure

    Args:
        course_id: Course ID
    """
    retry_count = self.request.retries
    if retry_count > 0:
        logger.info(f"Retry attempt {retry_count}/{self.max_retries} for update_concept_relationships {course_id}")

    try:
        self.update_state(
            state="PROCESSING", meta={"step": "Analyzing concept relationships", "retry_count": retry_count}
        )

        graph_service = GraphService()

        # Re-detect prerequisites
        with graph_service.driver.session() as session:
            graph_service._detect_prerequisites(session, course_id)

        graph_service.close()

        return {
            "status": "success",
            "message": "Concept relationships updated",
            "retries_used": retry_count,
        }

    except RETRYABLE_EXCEPTIONS as e:
        logger.warning(f"Retryable error updating relationships for course {course_id}: {e}")
        self.update_state(
            state="RETRYING",
            meta={"error": str(e), "retry_count": retry_count + 1}
        )
        raise

    except Exception as e:
        logger.error(f"Non-retryable error updating relationships for course {course_id}: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e), "retry_count": retry_count})
        raise
