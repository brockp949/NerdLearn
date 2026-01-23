"""
Worker configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Database URLs
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nerdlearn:password@localhost:5432/nerdlearn")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


    # MinIO/S3 Configuration
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "nerdlearn")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))  # tokens per chunk
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # overlap tokens
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large

    # Vector Store
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "course_chunks")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))  # OpenAI embedding size


config = Config()
