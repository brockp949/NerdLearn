import boto3
from botocore.client import Config
from app.core.config import settings
from typing import BinaryIO
import logging

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = settings.MINIO_BUCKET

    def _get_client(self):
        """Lazy initialization of S3 client"""
        if not self.s3_client:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
                aws_access_key_id=settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=settings.MINIO_SECRET_KEY,
                config=Config(signature_version='s3v4'),
                region_name='us-east-1'
            )
            # Ensure bucket exists
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
            except:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")

        return self.s3_client

    async def upload_file(
        self,
        file: BinaryIO,
        file_key: str,
        content_type: str = None
    ) -> str:
        """Upload a file to MinIO/S3 and return the file URL"""
        client = self._get_client()

        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type

        try:
            client.upload_fileobj(
                file,
                self.bucket_name,
                file_key,
                ExtraArgs=extra_args
            )

            # Generate file URL
            file_url = f"http://{settings.MINIO_ENDPOINT}/{self.bucket_name}/{file_key}"
            logger.info(f"Uploaded file: {file_url}")

            return file_url

        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise

    async def delete_file(self, file_key: str) -> bool:
        """Delete a file from MinIO/S3"""
        client = self._get_client()

        try:
            client.delete_object(Bucket=self.bucket_name, Key=file_key)
            logger.info(f"Deleted file: {file_key}")
            return True
        except Exception as e:
            logger.error(f"File deletion failed: {str(e)}")
            return False

    async def get_presigned_url(self, file_key: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for temporary file access"""
        client = self._get_client()

        try:
            url = client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Presigned URL generation failed: {str(e)}")
            raise


# Global instance
storage_service = StorageService()
