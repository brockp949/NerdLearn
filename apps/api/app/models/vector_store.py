from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from app.core.database import Base

class CourseChunk(Base):
    __tablename__ = "course_chunks"

    id = Column(String, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), index=True)
    module_id = Column(Integer, ForeignKey("modules.id"), nullable=True, index=True)
    
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
    
    module_type = Column(String)
    page_number = Column(Integer)
    heading = Column(String)
    
    meta_data = Column("metadata", JSONB, default={})
    
    timestamp_start = Column(Float, nullable=True)
    timestamp_end = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
