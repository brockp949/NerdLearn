"""
MinIO Service
Retrieves uploaded files from MinIO storage
"""
from minio import Minio
from minio.error import S3Error
from ..config import config
from io import BytesIO


class MinIOService:
    """Manages file retrieval from MinIO"""

    def __init__(self):
        self.client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE,
        )
        self.bucket = config.MINIO_BUCKET

    def get_file(self, object_name: str) -> bytes:
        """
        Retrieve a file from MinIO

        Args:
            object_name: Object path in MinIO

        Returns:
            File bytes

        Raises:
            Exception: If file not found or error occurs
        """
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            raise Exception(f"Failed to retrieve file from MinIO: {str(e)}")

    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in MinIO"""
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False

    def get_file_info(self, object_name: str) -> dict:
        """Get file metadata from MinIO"""
        try:
            stat = self.client.stat_object(self.bucket, object_name)
            return {
                "size": stat.size,
                "last_modified": stat.last_modified,
                "content_type": stat.content_type,
                "etag": stat.etag,
            }
        except S3Error as e:
            raise Exception(f"Failed to get file info: {str(e)}")
