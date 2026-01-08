"""
PDF Processing Tasks
"""
from ..celery_app import app
from ..processors.pdf_processor import PDFProcessor
from ..processors.chunker import SemanticChunker
from ..services.minio_service import MinIOService
from ..services.vector_store import VectorStoreService
from ..services.graph_service import GraphService
from ..config import config


@app.task(bind=True, name="process_pdf")
def process_pdf(self, module_id: int, course_id: int, file_path: str, title: str):
    """
    Process a PDF module

    Args:
        module_id: Module ID
        course_id: Course ID
        file_path: Path to file in MinIO
        title: Module title
    """
    try:
        # Update task state
        self.update_state(state="PROCESSING", meta={"step": "Downloading PDF"})

        # Download file from MinIO
        minio_service = MinIOService()
        file_bytes = minio_service.get_file(file_path)

        # Process PDF
        self.update_state(state="PROCESSING", meta={"step": "Extracting text"})
        pdf_processor = PDFProcessor()
        pdf_data = pdf_processor.process(file_bytes)

        # Chunk the content
        self.update_state(state="PROCESSING", meta={"step": "Chunking content"})
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
            state="PROCESSING", meta={"step": f"Storing {len(chunks)} chunks"}
        )
        vector_store = VectorStoreService()
        point_ids = vector_store.store_chunks(chunks, course_id, module_id, "pdf")

        # Extract concepts for knowledge graph
        self.update_state(state="PROCESSING", meta={"step": "Extracting concepts"})
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
        }

    except Exception as e:
        # Log error and fail
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
