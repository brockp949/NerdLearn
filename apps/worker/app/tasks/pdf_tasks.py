"""
PDF Processing Tasks with Exponential Backoff Retry
"""
import logging
from ..celery_app import app, DEFAULT_RETRY_KWARGS, RETRYABLE_EXCEPTIONS
from ..processors.pdf_processor import PDFProcessor
from ..processors.chunker import SemanticChunker
from ..services.minio_service import MinIOService
from ..services.vector_store import VectorStoreService
from ..services.graph_service import GraphService
from ..config import config

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="process_pdf",
    **DEFAULT_RETRY_KWARGS,
)
def process_pdf(self, module_id: int, course_id: int, file_path: str, title: str):
    """
    Process a PDF module with automatic retry on transient failures.

    Retry behavior:
    - Retries on: ConnectionError, TimeoutError, OSError, IOError
    - Exponential backoff: 2s, 4s, 8s, 16s, 32s (capped at 600s)
    - Jitter added to prevent thundering herd
    - Max 5 retries before permanent failure

    Args:
        module_id: Module ID
        course_id: Course ID
        file_path: Path to file in MinIO
        title: Module title
    """
    retry_count = self.request.retries
    if retry_count > 0:
        logger.info(f"Retry attempt {retry_count}/{self.max_retries} for PDF {module_id}")

    try:
        # Update task state with retry info
        meta = {"step": "Downloading PDF", "retry_count": retry_count}
        self.update_state(state="PROCESSING", meta=meta)

        # Download file from MinIO
        minio_service = MinIOService()
        file_bytes = minio_service.get_file(file_path)

        # Process PDF
        self.update_state(state="PROCESSING", meta={"step": "Extracting text", "retry_count": retry_count})
        pdf_processor = PDFProcessor()
        pdf_data = pdf_processor.process(file_bytes)

        # Chunk the content
        self.update_state(state="PROCESSING", meta={"step": "Chunking content", "retry_count": retry_count})
        chunker = SemanticChunker(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )

        doc_metadata = {
            "module_id": module_id,
            "course_id": course_id,
            "title": title,
            "type": "pdf",
        }

        chunks = chunker.chunk_document_pages(pdf_data["pages"], doc_metadata)

        # Store in vector database
        self.update_state(
            state="PROCESSING", meta={"step": f"Storing {len(chunks)} chunks", "retry_count": retry_count}
        )
        vector_store = VectorStoreService()
        point_ids = vector_store.store_chunks(chunks, course_id, module_id, "pdf")

        # Extract concepts for knowledge graph
        self.update_state(state="PROCESSING", meta={"step": "Extracting concepts", "retry_count": retry_count})
        graph_service = GraphService()
        concepts = graph_service.extract_concepts(pdf_data["full_text"])

        # Close graph connection
        graph_service.close()

        # Return results
        return {
            "status": "success",
            "module_id": module_id,
            "course_id": course_id,
            "statistics": pdf_data["statistics"],
            "chunk_count": len(chunks),
            "point_ids": point_ids,
            "concepts": concepts,
            "retries_used": retry_count,
        }

    except RETRYABLE_EXCEPTIONS as e:
        # Let Celery handle automatic retry for these exceptions
        logger.warning(f"Retryable error processing PDF {module_id}: {e}")
        self.update_state(
            state="RETRYING",
            meta={"error": str(e), "retry_count": retry_count + 1}
        )
        raise

    except Exception as e:
        # Non-retryable error - fail permanently
        logger.error(f"Non-retryable error processing PDF {module_id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "retry_count": retry_count}
        )
        raise
