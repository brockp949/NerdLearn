from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class CourseStatus(enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ModuleType(enum.Enum):
    VIDEO = "video"
    PDF = "pdf"
    QUIZ = "quiz"
    INTERACTIVE = "interactive"


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    description = Column(Text)
    instructor_id = Column(Integer, ForeignKey("instructors.id"), nullable=False)
    thumbnail_url = Column(String)
    status = Column(Enum(CourseStatus), default=CourseStatus.DRAFT)
    price = Column(Float, default=0.0)
    duration_hours = Column(Float)
    difficulty_level = Column(String)  # Beginner, Intermediate, Advanced
    tags = Column(String)  # JSON array of tags
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    published_at = Column(DateTime(timezone=True))

    # Relationships
    instructor = relationship("Instructor", back_populates="courses")
    modules = relationship("Module", back_populates="course", cascade="all, delete-orphan")
    enrollments = relationship("Enrollment", back_populates="course")


class Module(Base):
    __tablename__ = "modules"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    module_type = Column(Enum(ModuleType), nullable=False)
    order = Column(Integer, nullable=False)  # Display order in course
    duration_minutes = Column(Integer)

    # File storage
    file_url = Column(String)  # S3/MinIO URL
    file_size = Column(Integer)  # Size in bytes

    # Processing status
    is_processed = Column(Boolean, default=False)
    transcript_url = Column(String)  # For videos

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    course = relationship("Course", back_populates="modules")


class Enrollment(Base):
    __tablename__ = "enrollments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    progress_percentage = Column(Float, default=0.0)
    last_accessed_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")
