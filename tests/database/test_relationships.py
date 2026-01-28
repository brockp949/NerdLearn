"""
Relationship and Cascade Tests

Tests to verify database relationships and cascading behaviors work correctly.
Ensures referential integrity is maintained and cascade operations propagate
as expected.

Test Categories:
- One-to-Many relationships
- Many-to-Many relationships
- Cascade delete operations
- Orphan record handling
- Relationship loading (eager/lazy)
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .conftest import (
    UserFactory, CourseFactory, User, Course, Enrollment, Concept,
    SpacedRepetitionCard, ReviewLog, UserAchievement, UserStats,
    Instructor, TestBase
)

pytestmark = [pytest.mark.asyncio]


class TestUserRelationships:
    """Test User entity relationships."""

    async def test_user_has_enrollments(self, async_session: AsyncSession):
        """Verify user can have multiple enrollments."""
        user = User(**UserFactory.create())
        course1 = Course(**CourseFactory.create())
        course2 = Course(**CourseFactory.create())
        async_session.add_all([user, course1, course2])
        await async_session.commit()

        # Create enrollments
        enrollment1 = Enrollment(user_id=user.id, course_id=course1.id, progress=0.0)
        enrollment2 = Enrollment(user_id=user.id, course_id=course2.id, progress=0.0)
        async_session.add_all([enrollment1, enrollment2])
        await async_session.commit()

        # Capture ID before expire
        user_id = user.id

        # Load user with enrollments
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.enrollments))
            .where(User.id == user_id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.enrollments) == 2

    async def test_user_has_achievements(self, async_session: AsyncSession):
        """Verify user can have multiple achievements."""
        user = User(**UserFactory.create())
        async_session.add(user)
        await async_session.commit()

        achievements = [
            UserAchievement(
                user_id=user.id,
                achievement_type="STREAK_MILESTONE",
                name=f"Achievement {i}",
                description="Test",
                xp_reward=100
            )
            for i in range(3)
        ]
        async_session.add_all(achievements)
        await async_session.commit()

        # Capture ID before expire
        user_id = user.id

        # Load user with achievements
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.achievements))
            .where(User.id == user_id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.achievements) == 3

    async def test_user_has_spaced_repetition_cards(
        self, async_session: AsyncSession
    ):
        """Verify user can have multiple spaced repetition cards."""
        user = User(**UserFactory.create())
        concepts = [
            Concept(name=f"Concept {i}", description="Test")
            for i in range(3)
        ]
        async_session.add(user)
        async_session.add_all(concepts)
        await async_session.commit()

        cards = [
            SpacedRepetitionCard(
                user_id=user.id,
                concept_id=concept.id,
                difficulty=5.0,
                stability=2.5,
                retrievability=0.9,
                review_count=0,
                next_review_at=datetime.utcnow() + timedelta(days=1)
            )
            for concept in concepts
        ]
        async_session.add_all(cards)
        await async_session.commit()

        # Capture ID before expire
        user_id = user.id

        # Load user with cards
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.spaced_repetition_cards))
            .where(User.id == user_id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.spaced_repetition_cards) == 3


class TestCourseRelationships:
    """Test Course entity relationships."""

    async def test_course_has_enrollments(self, async_session: AsyncSession):
        """Verify course can have multiple enrollments."""
        course = Course(**CourseFactory.create())
        users = [User(**UserFactory.create()) for _ in range(3)]
        async_session.add(course)
        async_session.add_all(users)
        await async_session.commit()

        enrollments = [
            Enrollment(user_id=user.id, course_id=course.id, progress=0.0)
            for user in users
        ]
        async_session.add_all(enrollments)
        await async_session.commit()

        # Capture ID before expire
        course_id = course.id

        # Load course with enrollments
        result = await async_session.execute(
            select(Course)
            .options(selectinload(Course.enrollments))
            .where(Course.id == course_id)
        )
        loaded_course = result.scalar_one()

        assert len(loaded_course.enrollments) == 3

    async def test_course_has_instructor(self, async_session: AsyncSession):
        """Verify course can have an instructor."""
        user = User(**UserFactory.create(is_instructor=True))
        async_session.add(user)
        await async_session.commit()

        instructor = Instructor(
            user_id=user.id,
            bio="Test bio",
            expertise_areas="Python, SQL"
        )
        async_session.add(instructor)
        await async_session.commit()

        course = Course(**CourseFactory.create(instructor_id=instructor.id))
        async_session.add(course)
        await async_session.commit()

        # Capture ID before expire
        course_id = course.id

        # Load course with instructor
        result = await async_session.execute(
            select(Course)
            .options(selectinload(Course.instructor))
            .where(Course.id == course_id)
        )
        loaded_course = result.scalar_one()

        assert loaded_course.instructor is not None
        assert loaded_course.instructor_id == instructor.id


class TestCascadeDelete:
    """Test cascade delete operations.

    Note: These tests use ORM-level deletes (session.delete) because
    SQLite doesn't enforce ON DELETE CASCADE by default, and our models
    use SQLAlchemy ORM cascade which requires object-level deletion.
    """

    async def test_delete_user_cascades_to_enrollments(
        self, async_session: AsyncSession
    ):
        """Verify deleting user removes their enrollments via ORM cascade."""
        user = User(**UserFactory.create())
        course = Course(**CourseFactory.create())
        async_session.add_all([user, course])
        await async_session.commit()

        enrollment = Enrollment(user_id=user.id, course_id=course.id, progress=0.5)
        async_session.add(enrollment)
        await async_session.commit()
        enrollment_id = enrollment.id

        # Reload user to ensure it's attached to session
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.enrollments))
            .where(User.id == user.id)
        )
        user_to_delete = result.scalar_one()

        # Delete user via ORM (triggers cascade)
        await async_session.delete(user_to_delete)
        await async_session.commit()

        # Verify enrollment was deleted
        result = await async_session.execute(
            select(Enrollment).where(Enrollment.id == enrollment_id)
        )
        deleted_enrollment = result.scalar_one_or_none()

        assert deleted_enrollment is None

    async def test_delete_user_cascades_to_achievements(
        self, async_session: AsyncSession
    ):
        """Verify deleting user removes their achievements via ORM cascade."""
        user = User(**UserFactory.create())
        async_session.add(user)
        await async_session.commit()

        achievement = UserAchievement(
            user_id=user.id,
            achievement_type="STREAK_MILESTONE",
            name="Test",
            description="Test",
            xp_reward=100
        )
        async_session.add(achievement)
        await async_session.commit()
        achievement_id = achievement.id

        # Reload user with relationships
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.achievements))
            .where(User.id == user.id)
        )
        user_to_delete = result.scalar_one()

        # Delete user via ORM
        await async_session.delete(user_to_delete)
        await async_session.commit()

        # Verify achievement was deleted
        result = await async_session.execute(
            select(UserAchievement).where(UserAchievement.id == achievement_id)
        )
        deleted_achievement = result.scalar_one_or_none()

        assert deleted_achievement is None

    async def test_delete_user_cascades_to_spaced_repetition(
        self, async_session: AsyncSession
    ):
        """Verify deleting user removes their spaced repetition cards via ORM cascade."""
        user = User(**UserFactory.create())
        concept = Concept(name="Test", description="Test")
        async_session.add_all([user, concept])
        await async_session.commit()

        card = SpacedRepetitionCard(
            user_id=user.id,
            concept_id=concept.id,
            difficulty=5.0,
            stability=2.5,
            retrievability=0.9,
            review_count=0,
            next_review_at=datetime.utcnow() + timedelta(days=1)
        )
        async_session.add(card)
        await async_session.commit()

        review = ReviewLog(
            card_id=card.id,
            rating=4,
            reviewed_at=datetime.utcnow(),
            response_time_ms=2000
        )
        async_session.add(review)
        await async_session.commit()
        card_id = card.id
        review_id = review.id

        # Reload user with relationships
        result = await async_session.execute(
            select(User)
            .options(
                selectinload(User.spaced_repetition_cards)
                .selectinload(SpacedRepetitionCard.review_logs)
            )
            .where(User.id == user.id)
        )
        user_to_delete = result.scalar_one()

        # Delete user via ORM
        await async_session.delete(user_to_delete)
        await async_session.commit()

        # Verify card was deleted
        result = await async_session.execute(
            select(SpacedRepetitionCard).where(SpacedRepetitionCard.id == card_id)
        )
        assert result.scalar_one_or_none() is None

        # Verify review log was also deleted (cascade through card)
        result = await async_session.execute(
            select(ReviewLog).where(ReviewLog.id == review_id)
        )
        assert result.scalar_one_or_none() is None

    async def test_delete_course_cascades_to_enrollments(
        self, async_session: AsyncSession
    ):
        """Verify deleting course removes all enrollments via ORM cascade."""
        course = Course(**CourseFactory.create())
        users = [User(**UserFactory.create()) for _ in range(3)]
        async_session.add(course)
        async_session.add_all(users)
        await async_session.commit()

        enrollments = [
            Enrollment(user_id=user.id, course_id=course.id, progress=0.0)
            for user in users
        ]
        async_session.add_all(enrollments)
        await async_session.commit()
        course_id = course.id

        # Reload course with relationships
        result = await async_session.execute(
            select(Course)
            .options(selectinload(Course.enrollments))
            .where(Course.id == course_id)
        )
        course_to_delete = result.scalar_one()

        # Delete course via ORM
        await async_session.delete(course_to_delete)
        await async_session.commit()

        # Verify all enrollments were deleted
        result = await async_session.execute(
            select(Enrollment).where(Enrollment.course_id == course_id)
        )
        remaining_enrollments = result.scalars().all()

        assert len(remaining_enrollments) == 0

    async def test_delete_card_cascades_to_review_logs(
        self, async_session: AsyncSession
    ):
        """Verify deleting a spaced repetition card removes its review logs via ORM cascade."""
        user = User(**UserFactory.create())
        concept = Concept(name="Test", description="Test")
        async_session.add_all([user, concept])
        await async_session.commit()

        card = SpacedRepetitionCard(
            user_id=user.id,
            concept_id=concept.id,
            difficulty=5.0,
            stability=2.5,
            retrievability=0.9,
            review_count=0,
            next_review_at=datetime.utcnow() + timedelta(days=1)
        )
        async_session.add(card)
        await async_session.commit()

        # Create multiple review logs
        reviews = [
            ReviewLog(
                card_id=card.id,
                rating=i,
                reviewed_at=datetime.utcnow(),
                response_time_ms=2000
            )
            for i in range(1, 5)
        ]
        async_session.add_all(reviews)
        await async_session.commit()
        card_id = card.id

        # Reload card with relationships
        result = await async_session.execute(
            select(SpacedRepetitionCard)
            .options(selectinload(SpacedRepetitionCard.review_logs))
            .where(SpacedRepetitionCard.id == card_id)
        )
        card_to_delete = result.scalar_one()

        # Delete card via ORM
        await async_session.delete(card_to_delete)
        await async_session.commit()

        # Verify all review logs were deleted
        result = await async_session.execute(
            select(ReviewLog).where(ReviewLog.card_id == card_id)
        )
        remaining_logs = result.scalars().all()

        assert len(remaining_logs) == 0


class TestOrphanRecords:
    """Test handling of orphaned records."""

    async def test_delete_enrollment_preserves_user(
        self, async_session: AsyncSession
    ):
        """Verify deleting enrollment does not affect user."""
        user = User(**UserFactory.create())
        course = Course(**CourseFactory.create())
        async_session.add_all([user, course])
        await async_session.commit()
        user_id = user.id

        enrollment = Enrollment(user_id=user.id, course_id=course.id, progress=0.5)
        async_session.add(enrollment)
        await async_session.commit()

        # Delete enrollment
        await async_session.execute(
            delete(Enrollment).where(Enrollment.user_id == user_id)
        )
        await async_session.commit()

        # Verify user still exists
        result = await async_session.execute(
            select(User).where(User.id == user_id)
        )
        existing_user = result.scalar_one_or_none()

        assert existing_user is not None

    async def test_delete_enrollment_preserves_course(
        self, async_session: AsyncSession
    ):
        """Verify deleting enrollment does not affect course."""
        user = User(**UserFactory.create())
        course = Course(**CourseFactory.create())
        async_session.add_all([user, course])
        await async_session.commit()
        course_id = course.id

        enrollment = Enrollment(user_id=user.id, course_id=course.id, progress=0.5)
        async_session.add(enrollment)
        await async_session.commit()

        # Delete enrollment
        await async_session.execute(
            delete(Enrollment).where(Enrollment.course_id == course_id)
        )
        await async_session.commit()

        # Verify course still exists
        result = await async_session.execute(
            select(Course).where(Course.id == course_id)
        )
        existing_course = result.scalar_one_or_none()

        assert existing_course is not None


class TestRelationshipLoading:
    """Test relationship loading strategies."""

    async def test_eager_load_enrollments(self, async_session: AsyncSession):
        """Test eager loading of enrollments with selectinload."""
        user = User(**UserFactory.create())
        courses = [Course(**CourseFactory.create()) for _ in range(3)]
        async_session.add(user)
        async_session.add_all(courses)
        await async_session.commit()

        # Capture user_id before creating enrollments
        user_id = user.id

        enrollments = [
            Enrollment(user_id=user_id, course_id=course.id, progress=0.0)
            for course in courses
        ]
        async_session.add_all(enrollments)
        await async_session.commit()

        async_session.expire_all()

        # Load with eager loading - use captured user_id
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.enrollments).selectinload(Enrollment.course))
            .where(User.id == user_id)
        )
        loaded_user = result.scalar_one()

        # Verify enrollments and courses are loaded
        assert len(loaded_user.enrollments) == 3
        for enrollment in loaded_user.enrollments:
            assert enrollment.course is not None

    async def test_nested_relationship_loading(self, async_session: AsyncSession):
        """Test loading nested relationships."""
        user = User(**UserFactory.create())
        concept = Concept(name="Test", description="Test")
        async_session.add_all([user, concept])
        await async_session.commit()

        # Capture user_id before further operations
        user_id = user.id

        card = SpacedRepetitionCard(
            user_id=user_id,
            concept_id=concept.id,
            difficulty=5.0,
            stability=2.5,
            retrievability=0.9,
            review_count=0,
            next_review_at=datetime.utcnow() + timedelta(days=1)
        )
        async_session.add(card)
        await async_session.commit()

        reviews = [
            ReviewLog(
                card_id=card.id,
                rating=i,
                reviewed_at=datetime.utcnow(),
                response_time_ms=2000
            )
            for i in range(1, 4)
        ]
        async_session.add_all(reviews)
        await async_session.commit()

        async_session.expire_all()

        # Load user with nested relationships - use captured user_id
        result = await async_session.execute(
            select(User)
            .options(
                selectinload(User.spaced_repetition_cards)
                .selectinload(SpacedRepetitionCard.review_logs)
            )
            .where(User.id == user_id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.spaced_repetition_cards) == 1
        assert len(loaded_user.spaced_repetition_cards[0].review_logs) == 3
