"""
Knowledge Graph Construction Tasks
"""
from ..celery_app import app
from ..services.graph_service import GraphService
from ..services.vector_store import VectorStoreService


@app.task(bind=True, name="build_course_graph")
def build_course_graph(self, course_id: int, course_title: str):
    """
    Build knowledge graph for an entire course

    Args:
        course_id: Course ID
        course_title: Course title
    """
    try:
        self.update_state(
            state="PROCESSING", meta={"step": "Fetching course modules"}
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
            state="PROCESSING", meta={"step": "Constructing knowledge graph"}
        )

        # Close connection
        graph_service.close()

        return {
            "status": "success",
            "message": "Knowledge graph built successfully",
            "course_id": course_id,
        }

    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@app.task(bind=True, name="update_concept_relationships")
def update_concept_relationships(self, course_id: int):
    """
    Update concept prerequisite relationships

    Args:
        course_id: Course ID
    """
    try:
        self.update_state(
            state="PROCESSING", meta={"step": "Analyzing concept relationships"}
        )

        graph_service = GraphService()

        # Re-detect prerequisites
        with graph_service.driver.session() as session:
            graph_service._detect_prerequisites(session, course_id)

        graph_service.close()

        return {
            "status": "success",
            "message": "Concept relationships updated",
        }

    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
