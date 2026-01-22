from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class CourseStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ModuleType(str, Enum):
    VIDEO = "video"
    PDF = "pdf"
    QUIZ = "quiz"
    INTERACTIVE = "interactive"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Course Schemas
class CourseBase(BaseModel):
    title: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    price: float = 0.0
    difficulty_level: Optional[str] = None
    tags: Optional[List[str]] = []


class CourseCreate(CourseBase):
    instructor_id: int


class CourseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    status: Optional[CourseStatus] = None
    price: Optional[float] = None
    difficulty_level: Optional[str] = None
    tags: Optional[List[str]] = None


class CourseResponse(CourseBase):
    id: int
    instructor_id: int
    status: CourseStatus
    duration_hours: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Module Schemas
class ModuleBase(BaseModel):
    title: str
    description: Optional[str] = None
    module_type: ModuleType
    order: int
    duration_minutes: Optional[int] = None


class ModuleCreate(ModuleBase):
    course_id: int


class ModuleUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = None
    duration_minutes: Optional[int] = None


class ModuleResponse(ModuleBase):
    id: int
    course_id: int
    file_url: Optional[str] = None
    file_size: Optional[int] = None
    is_processed: bool = False
    processing_status: Optional[ProcessingStatus] = ProcessingStatus.PENDING
    processing_task_id: Optional[str] = None
    processing_error: Optional[str] = None
    transcript_url: Optional[str] = None
    chunk_count: Optional[int] = 0
    concept_count: Optional[int] = 0
    processed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ModuleUploadResponse(BaseModel):
    module_id: int
    upload_url: str
    message: str
