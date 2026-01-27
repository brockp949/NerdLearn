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

from .conftest import UserFactory, CourseFactory

pytestmark = [pytest.mark.requires_db, pytest.mark.unit]


class TestUserRelationships:
    """Test User entity relationships."""

    @pytest.mark.asyncio
    async def test_user_has_enrollments(self, async_session: AsyncSession):
        """Verify user can have multiple enrollments."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

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

        # Load user with enrollments
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.enrollments))
            .where(User.id == user.id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.enrollments) == 2

    @pytest.mark.asyncio
    async def test_user_has_achievements(self, async_session: AsyncSession):
        """Verify user can have multiple achievements."""
        from app.models.user import User
        from app.models.gamification import UserAchievement

        user = User(**UserFactory.create())
        async_session.add(user)
        await async_session.commit()

        achievements = [
            UserAchievement(
                user_id=user.id,
                achievement_type="STREAK_MILESTONE",
                name=f"Achievement {i}",
                description=f"Description {i}",
                xp_reward=100
            )
            for i in range(3)
        ]
        async_session.add_all(achievements)
        await async_session.commit()

        result = await async_session.execute(
            select(User)
            .options(selectinload(User.achievements))
            .where(User.id == user.id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.achievements) == 3

    @pytest.mark.asyncio
    async def test_user_has_spaced_repetition_cards(self, async_session: AsyncSession):
        """Verify user can have multiple spaced repetition cards."""
        from app.models.user import User
        from app.models.spaced_repetition import SpacedRepetitionCard, Concept

        user = User(**UserFactory.create())
        async_session.add(user)
        await async_session.commit()

        # Create concepts and cards
        for i in range(5):
            concept = Concept(name=f"Concept {i}", description=f"Description {i}")
            async_session.add(concept)
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

        result = await async_session.execute(
            select(User)
            .options(selectinload(User.spaced_repetition_cards))
            .where(User.id == user.id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.spaced_repetition_cards) == 5


class TestCourseRelationships:
    """Test Course entity relationships."""

    @pytest.mark.asyncio
    async def test_course_has_enrollments(self, async_session: AsyncSession):
        """Verify course can have multiple enrolled students."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

        course = Course(**CourseFactory.create())
        users = [User(**UserFactory.create()) for _ in range(5)]
        async_session.add(course)
        async_session.add_all(users)
        await async_session.commit()

        enrollments = [
            Enrollment(user_id=user.id, course_id=course.id, progress=0.0)
            for user in users
        ]
        async_session.add_all(enrollments)
        await async_session.commit()

        result = await async_session.execute(
            select(Course)
            .options(selectinload(Course.enrollments))
            .where(Course.id == course.id)
        )
        loaded_course = result.scalar_one()

        assert len(loaded_course.enrollments) == 5

    @pytest.mark.asyncio
    async def test_course_has_instructor(self, async_session: AsyncSession):
        """Verify course can be assigned to an instructor."""
        from app.models.user import User, Instructor
        from app.models.course import Course

        user = User(**UserFactory.create(is_instructor=True))
        async_session.add(user)
        await async_session.commit()

        instructor = Instructor(
            user_id=user.id,
            bio="Test instructor",
            expertise_areas="Python, AI"
        )
        async_session.add(instructor)
        await async_session.commit()

        course = Course(**CourseFactory.create(instructor_id=instructor.id))
        async_session.add(course)
        await async_session.commit()

        result = await async_session.execute(
            select(Course)
            .options(selectinload(Course.instructor))
            .where(Course.id == course.id)
        )
        loaded_course = result.scalar_one()

        assert loaded_course.instructor_id == instructor.id


class TestCascadeDelete:
    """Test cascade delete operations."""

    @pytest.mark.asyncio
    async def test_delete_user_cascades_to_enrollments(
        self, async_session: AsyncSession
    ):
        """Verify deleting user removes their enrollments."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

        user = User(**UserFactory.create())
        course = Course(**CourseFactory.create())
        async_session.add_all([user, course])
        await async_session.commit()

        enrollment = Enrollment(user_id=user.id, course_id=course.id, progress=0.5)
        async_session.add(enrollment)
        await async_session.commit()
        enrollment_id = enrollment.id

        # Delete user
        await async_session.execute(
            delete(User).where(User.id == user.id)
        )
        await async_session.commit()

        # Verify enrollment was deleted
        result = await async_session.execute(
            select(Enrollment).where(Enrollment.id == enrollment_id)
        )
        deleted_enrollment = result.scalar_one_or_none()

        assert deleted_enrollment is None

    @pytest.mark.asyncio
    async def test_delete_user_cascades_to_achievements(
        self, async_session: AsyncSession
    ):
        """Verify deleting user removes their achievements."""
        from app.models.user import User
        from app.models.gamification import UserAchievement

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

        # Delete user
        await async_session.execute(
            delete(User).where(User.id == user.id)
        )
        await async_session.commit()

        # Verify achievement was deleted
        result = await async_session.execute(
            select(UserAchievement).where(UserAchievement.id == achievement_id)
        )
        deleted_achievement = result.scalar_one_or_none()

        assert deleted_achievement is None

    @pytest.mark.asyncio
    async def test_delete_user_cascades_to_spaced_repetition(
        self, async_session: AsyncSession
    ):
        """Verify deleting user removes their spaced repetition cards."""
        from app.models.user import User
        from app.models.spaced_repetition import SpacedRepetitionCard, Concept, ReviewLog

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

        # Delete user
        await async_session.execute(
            delete(User).where(User.id == user.id)
        )
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

    @pytest.mark.asyncio
    async def test_delete_course_cascades_to_enrollments(
        self, async_session: AsyncSession
    ):
        """Verify deleting course removes all enrollments."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

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

        # Delete course
        await async_session.execute(
            delete(Course).where(Course.id == course_id)
        )
        await async_session.commit()

        # Verify all enrollments were deleted
        result = await async_session.execute(
            select(Enrollment).where(Enrollment.course_id == course_id)
        )
        remaining_enrollments = result.scalars().all()

        assert len(remaining_enrollments) == 0

    @pytest.mark.asyncio
    async def test_delete_card_cascades_to_review_logs(
        self, async_session: AsyncSession
    ):
        """Verify deleting a spaced repetition card removes its review logs."""
        from app.models.user import User
        from app.models.spaced_repetition import SpacedRepetitionCard, Concept, ReviewLog

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

        # Delete card
        await async_session.execute(
            delete(SpacedRepetitionCard).where(SpacedRepetitionCard.id == card_id)
        )
        await async_session.commit()

        # Verify all review logs were deleted
        result = await async_session.execute(
            select(ReviewLog).where(ReviewLog.card_id == card_id)
        )
        remaining_logs = result.scalars().all()

        assert len(remaining_logs) == 0


class TestOrphanRecords:
    """Test handling of orphaned records."""

    @pytest.mark.asyncio
    async def test_delete_enrollment_preserves_user(
        self, async_session: AsyncSession
    ):
        """Verify deleting enrollment does not affect user."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

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
        preserved_user = result.scalar_one_or_none()

        assert preserved_user is not None
        assert preserved_user.id == user_id

    @pytest.mark.asyncio
    async def test_delete_enrollment_preserves_course(
        self, async_session: AsyncSession
    ):
        """Verify deleting enrollment does not affect course."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

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
        preserved_course = result.scalar_one_or_none()

        assert preserved_course is not None


class TestRelationshipLoading:
    """Test relationship loading strategies."""

    @pytest.mark.asyncio
    async def test_eager_load_enrollments(self, async_session: AsyncSession):
        """Test eager loading of enrollments with selectinload."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

        user = User(**UserFactory.create())
        courses = [Course(**CourseFactory.create()) for _ in range(3)]
        async_session.add(user)
        async_session.add_all(courses)
        await async_session.commit()

        enrollments = [
            Enrollment(user_id=user.id, course_id=course.id, progress=0.0)
            for course in courses
        ]
        async_session.add_all(enrollments)
        await async_session.commit()

        async_session.expire_all()

        # Load with eager loading
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.enrollments).selectinload(Enrollment.course))
            .where(User.id == user.id)
        )
        loaded_user = result.scalar_one()

        # Verify enrollments and courses are loaded
        assert len(loaded_user.enrollments) == 3
        for enrollment in loaded_user.enrollments:
            assert enrollment.course is not None

    @pytest.mark.asyncio
    async def test_nested_relationship_loading(self, async_session: AsyncSession):
        """Test loading nested relationships."""
        from app.models.user import User
        from app.models.spaced_repetition import SpacedRepetitionCard, Concept, ReviewLog

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

        # Load user with nested relationships
        result = await async_session.execute(
            select(User)
            .options(
                selectinload(User.spaced_repetition_cards)
                .selectinload(SpacedRepetitionCard.review_logs)
            )
            .where(User.id == user.id)
        )
        loaded_user = result.scalar_one()

        assert len(loaded_user.spaced_repetition_cards) == 1
        assert len(loaded_user.spaced_repetition_cards[0].review_logs) == 3
