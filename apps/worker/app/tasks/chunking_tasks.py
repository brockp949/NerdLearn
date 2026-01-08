"""
Chunking and Embedding Tasks
"""
from ..celery_app import app
from ..services.vector_store import VectorStoreService


@app.task(bind=True, name="reindex_module")
def reindex_module(self, module_id: int, course_id: int):
    """
    Reindex a module (regenerate embeddings)

    Args:
        module_id: Module ID
        course_id: Course ID
    """
    try:
        self.update_state(state="PROCESSING", meta={"step": "Deleting old embeddings"})

        vector_store = VectorStoreService()

        # Delete existing chunks
        vector_store.delete_module_chunks(module_id)

        # Note: This would need to re-trigger processing based on module type
        # For now, we just delete. Full re-indexing would call process_pdf or process_video

        return {
            "status": "success",
            "message": "Module embeddings deleted. Re-upload module to reindex.",
        }

    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
