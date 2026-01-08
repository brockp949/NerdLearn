"""
Video Processing Tasks
"""
from ..celery_app import app
from ..processors.video_processor import VideoProcessor
from ..processors.chunker import SemanticChunker
from ..services.minio_service import MinIOService
from ..services.vector_store import VectorStoreService
from ..services.graph_service import GraphService
from ..config import config


@app.task(bind=True, name="process_video")
def process_video(self, module_id: int, course_id: int, file_path: str, title: str):
    """
    Process a video module (transcription)

    Args:
        module_id: Module ID
        course_id: Course ID
        file_path: Path to file in MinIO
        title: Module title
    """
    try:
        # Update task state
        self.update_state(state="PROCESSING", meta={"step": "Downloading video"})

        # Download file from MinIO
        minio_service = MinIOService()
        file_bytes = minio_service.get_file(file_path)

        # Extract filename from path
        filename = file_path.split("/")[-1]

        # Transcribe video
        self.update_state(state="PROCESSING", meta={"step": "Transcribing audio"})
        video_processor = VideoProcessor(model_name=config.WHISPER_MODEL)
        transcript_data = video_processor.process(file_bytes, filename)

        # Chunk the transcript
        self.update_state(state="PROCESSING", meta={"step": "Chunking transcript"})
        chunker = SemanticChunker(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )

        # Create chunks from segments
        full_text = transcript_data["text"]

        doc_metadata = {
            "module_id": module_id,
            "course_id": course_id,
            "title": title,
            "type": "video",
            "duration": transcript_data["metadata"]["duration"],
            "language": transcript_data["metadata"]["language"],
        }

        chunks = chunker.chunk_text(full_text, doc_metadata)

        # Enhance chunks with timestamp information
        for i, chunk in enumerate(chunks):
            # Find corresponding segment
            chunk_text = chunk["text"]
            for segment in transcript_data["segments"]:
                if segment["text"] in chunk_text:
                    chunk["metadata"]["timestamp_start"] = segment["start"]
                    chunk["metadata"]["timestamp_end"] = segment["end"]
                    break

        # Store in vector database
        self.update_state(
            state="PROCESSING", meta={"step": f"Storing {len(chunks)} chunks"}
        )
        vector_store = VectorStoreService()
        point_ids = vector_store.store_chunks(chunks, course_id, module_id, "video")

        # Extract concepts for knowledge graph
        self.update_state(state="PROCESSING", meta={"step": "Extracting concepts"})
        graph_service = GraphService()
        concepts = graph_service.extract_concepts(full_text)

        # Close graph connection
        graph_service.close()

        # Return results
        return {
            "status": "success",
            "module_id": module_id,
            "course_id": course_id,
            "statistics": transcript_data["statistics"],
            "chunk_count": len(chunks),
            "point_ids": point_ids,
            "concepts": concepts,
            "segments": len(transcript_data["segments"]),
            "transcript": transcript_data,
        }

    except Exception as e:
        # Log error and fail
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
