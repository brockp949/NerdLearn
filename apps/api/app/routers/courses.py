from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.core.database import get_db
from app.models.course import Course, CourseStatus
from app.schemas.course import CourseCreate, CourseUpdate, CourseResponse

router = APIRouter()


@router.get("/", response_model=List[CourseResponse])
async def list_courses(
    status_filter: str = None,
    instructor_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all courses with optional filtering"""
    query = select(Course)

    if status_filter:
        query = query.where(Course.status == CourseStatus(status_filter))

    if instructor_id:
        query = query.where(Course.instructor_id == instructor_id)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    courses = result.scalars().all()

    return courses


@router.post("/", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
async def create_course(course: CourseCreate, db: AsyncSession = Depends(get_db)):
    """Create a new course (Instructor Studio)"""
    db_course = Course(
        title=course.title,
        description=course.description,
        instructor_id=course.instructor_id,
        thumbnail_url=course.thumbnail_url,
        price=course.price,
        difficulty_level=course.difficulty_level,
        tags=",".join(course.tags) if course.tags else "",
        status=CourseStatus.DRAFT
    )

    db.add(db_course)
    await db.flush()
    await db.refresh(db_course)

    return db_course


@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(course_id: int, db: AsyncSession = Depends(get_db)):
    """Get course by ID"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    return course


@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: int,
    course_update: CourseUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update course details"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Update fields
    update_data = course_update.dict(exclude_unset=True)
    if "tags" in update_data and update_data["tags"]:
        update_data["tags"] = ",".join(update_data["tags"])

    for field, value in update_data.items():
        setattr(course, field, value)

    await db.flush()
    await db.refresh(course)

    return course


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(course_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a course"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    await db.delete(course)

    return None


@router.post("/{course_id}/publish", response_model=CourseResponse)
async def publish_course(course_id: int, db: AsyncSession = Depends(get_db)):
    """Publish a course (change status from draft to published)"""
    result = await db.execute(select(Course).where(Course.id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    course.status = CourseStatus.PUBLISHED
    from datetime import datetime
    course.published_at = datetime.now()

    await db.flush()
    await db.refresh(course)

    return course
