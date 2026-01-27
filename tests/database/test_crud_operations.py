"""
CRUD Operation Tests

Tests for Create, Read, Update, Delete operations across all major entities.
Verifies that basic database operations work correctly and data is persisted
as expected.

Test Categories:
- User CRUD operations
- Course CRUD operations
- Enrollment CRUD operations
- Spaced Repetition CRUD operations
- Achievement CRUD operations
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from .conftest import UserFactory, CourseFactory, ConceptFactory

pytestmark = [pytest.mark.requires_db, pytest.mark.unit]


class TestUserCRUD:
    """Test User entity CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_user(self, async_session: AsyncSession):
        """Test creating a new user."""
        from app.models.user import User

        user_data = UserFactory.create()
        user = User(**user_data)
        async_session.add(user)
        await async_session.commit()

        assert user.id is not None
        assert user.email == user_data["email"]
        assert user.username == user_data["username"]
        assert user.created_at is not None

    @pytest.mark.asyncio
    async def test_read_user_by_id(self, async_session: AsyncSession):
        """Test reading a user by ID."""
        from app.models.user import User

        # Create user
        user_data = UserFactory.create()
        user = User(**user_data)
        async_session.add(user)
        await async_session.commit()
        user_id = user.id

        # Clear session cache
        async_session.expire_all()

        # Read user
        result = await async_session.execute(
            select(User).where(User.id == user_id)
        )
        fetched_user = result.scalar_one()

        assert fetched_user.id == user_id
        assert fetched_user.email == user_data["email"]

    @pytest.mark.asyncio
    async def test_read_user_by_email(self, async_session: AsyncSession):
        """Test reading a user by email address."""
        from app.models.user import User

        user_data = UserFactory.create(email="findme@example.com")
        user = User(**user_data)
        async_session.add(user)
        await async_session.commit()

        async_session.expire_all()

        result = await async_session.execute(
            select(User).where(User.email == "findme@example.com")
        )
        fetched_user = result.scalar_one_or_none()

        assert fetched_user is not None
        assert fetched_user.email == "findme@example.com"

    @pytest.mark.asyncio
    async def test_update_user(self, async_session: AsyncSession):
        """Test updating user fields."""
        from app.models.user import User

        # Create user
        user_data = UserFactory.create()
        user = User(**user_data)
        async_session.add(user)
        await async_session.commit()
        user_id = user.id

        # Update user
        await async_session.execute(
            update(User)
            .where(User.id == user_id)
            .values(full_name="Updated Name", total_xp=500)
        )
        await async_session.commit()

        async_session.expire_all()

        # Verify update
        result = await async_session.execute(
            select(User).where(User.id == user_id)
        )
        updated_user = result.scalar_one()

        assert updated_user.full_name == "Updated Name"
        assert updated_user.total_xp == 500

    @pytest.mark.asyncio
    async def test_update_user_partial(self, async_session: AsyncSession):
        """Test partial update (only specific fields)."""
        from app.models.user import User

        user_data = UserFactory.create(total_xp=100, level=5)
        user = User(**user_data)
        async_session.add(user)
        await async_session.commit()
        user_id = user.id

        # Update only XP, level should remain
        await async_session.execute(
            update(User)
            .where(User.id == user_id)
            .values(total_xp=200)
        )
        await async_session.commit()

        async_session.expire_all()

        result = await async_session.execute(
            select(User).where(User.id == user_id)
        )
        updated_user = result.scalar_one()

        assert updated_user.total_xp == 200
        assert updated_user.level == 5  # Unchanged

    @pytest.mark.asyncio
    async def test_delete_user(self, async_session: AsyncSession):
        """Test deleting a user."""
        from app.models.user import User

        user_data = UserFactory.create()
        user = User(**user_data)
        async_session.add(user)
        await async_session.commit()
        user_id = user.id

        # Delete user
        await async_session.execute(
            delete(User).where(User.id == user_id)
        )
        await async_session.commit()

        # Verify deletion
        result = await async_session.execute(
            select(User).where(User.id == user_id)
        )
        deleted_user = result.scalar_one_or_none()

        assert deleted_user is None

    @pytest.mark.asyncio
    async def test_create_multiple_users(self, async_session: AsyncSession):
        """Test creating multiple users in batch."""
        from app.models.user import User

        users = [User(**UserFactory.create()) for _ in range(5)]
        async_session.add_all(users)
        await async_session.commit()

        # Verify all users created
        result = await async_session.execute(select(User))
        all_users = result.scalars().all()

        assert len(all_users) == 5


class TestCourseCRUD:
    """Test Course entity CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_course(self, async_session: AsyncSession):
        """Test creating a new course."""
        from app.models.course import Course

        course_data = CourseFactory.create()
        course = Course(**course_data)
        async_session.add(course)
        await async_session.commit()

        assert course.id is not None
        assert course.title == course_data["title"]

    @pytest.mark.asyncio
    async def test_read_courses_by_domain(self, async_session: AsyncSession):
        """Test reading courses filtered by domain."""
        from app.models.course import Course

        # Create courses in different domains
        course1 = Course(**CourseFactory.create(domain="Mathematics"))
        course2 = Course(**CourseFactory.create(domain="Computer Science"))
        course3 = Course(**CourseFactory.create(domain="Mathematics"))
        async_session.add_all([course1, course2, course3])
        await async_session.commit()

        # Query by domain
        result = await async_session.execute(
            select(Course).where(Course.domain == "Mathematics")
        )
        math_courses = result.scalars().all()

        assert len(math_courses) == 2

    @pytest.mark.asyncio
    async def test_update_course_publish_status(self, async_session: AsyncSession):
        """Test publishing a course."""
        from app.models.course import Course

        course = Course(**CourseFactory.create(is_published=False))
        async_session.add(course)
        await async_session.commit()
        course_id = course.id

        # Publish course
        await async_session.execute(
            update(Course)
            .where(Course.id == course_id)
            .values(is_published=True)
        )
        await async_session.commit()

        async_session.expire_all()

        result = await async_session.execute(
            select(Course).where(Course.id == course_id)
        )
        updated_course = result.scalar_one()

        assert updated_course.is_published is True

    @pytest.mark.asyncio
    async def test_delete_course(self, async_session: AsyncSession):
        """Test deleting a course."""
        from app.models.course import Course

        course = Course(**CourseFactory.create())
        async_session.add(course)
        await async_session.commit()
        course_id = course.id

        await async_session.execute(
            delete(Course).where(Course.id == course_id)
        )
        await async_session.commit()

        result = await async_session.execute(
            select(Course).where(Course.id == course_id)
        )
        assert result.scalar_one_or_none() is None


class TestEnrollmentCRUD:
    """Test Enrollment entity CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_enrollment(self, async_session: AsyncSession):
        """Test enrolling a user in a course."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

        user = User(**UserFactory.create())
        course = Course(**CourseFactory.create())
        async_session.add_all([user, course])
        await async_session.commit()

        enrollment = Enrollment(
            user_id=user.id,
            course_id=course.id,
            progress=0.0
        )
        async_session.add(enrollment)
        await async_session.commit()

        assert enrollment.id is not None
        assert enrollment.user_id == user.id
        assert enrollment.course_id == course.id

    @pytest.mark.asyncio
    async def test_update_enrollment_progress(self, async_session: AsyncSession):
        """Test updating enrollment progress."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

        user = User(**UserFactory.create())
        course = Course(**CourseFactory.create())
        async_session.add_all([user, course])
        await async_session.commit()

        enrollment = Enrollment(
            user_id=user.id,
            course_id=course.id,
            progress=0.0
        )
        async_session.add(enrollment)
        await async_session.commit()
        enrollment_id = enrollment.id

        # Update progress
        await async_session.execute(
            update(Enrollment)
            .where(Enrollment.id == enrollment_id)
            .values(progress=0.5)
        )
        await async_session.commit()

        async_session.expire_all()

        result = await async_session.execute(
            select(Enrollment).where(Enrollment.id == enrollment_id)
        )
        updated_enrollment = result.scalar_one()

        assert updated_enrollment.progress == 0.5

    @pytest.mark.asyncio
    async def test_complete_enrollment(self, async_session: AsyncSession):
        """Test marking enrollment as completed."""
        from app.models.user import User
        from app.models.course import Course, Enrollment

        user = User(**UserFactory.create())
        course = Course(**CourseFactory.create())
        async_session.add_all([user, course])
        await async_session.commit()

        enrollment = Enrollment(
            user_id=user.id,
            course_id=course.id,
            progress=0.0
        )
        async_session.add(enrollment)
        await async_session.commit()
        enrollment_id = enrollment.id

        # Complete enrollment
        completion_time = datetime.utcnow()
        await async_session.execute(
            update(Enrollment)
            .where(Enrollment.id == enrollment_id)
            .values(progress=1.0, completed_at=completion_time)
        )
        await async_session.commit()

        async_session.expire_all()

        result = await async_session.execute(
            select(Enrollment).where(Enrollment.id == enrollment_id)
        )
        completed_enrollment = result.scalar_one()

        assert completed_enrollment.progress == 1.0
        assert completed_enrollment.completed_at is not None


class TestSpacedRepetitionCRUD:
    """Test Spaced Repetition entities CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_spaced_repetition_card(self, async_session: AsyncSession):
        """Test creating a new spaced repetition card."""
        from app.models.user import User
        from app.models.spaced_repetition import SpacedRepetitionCard, Concept

        user = User(**UserFactory.create())
        concept = Concept(name="Test Concept", description="Test description")
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

        assert card.id is not None
        assert card.difficulty == 5.0

    @pytest.mark.asyncio
    async def test_update_fsrs_parameters(self, async_session: AsyncSession):
        """Test updating FSRS parameters after review."""
        from app.models.user import User
        from app.models.spaced_repetition import SpacedRepetitionCard, Concept

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
        card_id = card.id

        # Simulate FSRS update after review
        new_stability = 5.0
        new_difficulty = 4.5
        new_next_review = datetime.utcnow() + timedelta(days=7)

        await async_session.execute(
            update(SpacedRepetitionCard)
            .where(SpacedRepetitionCard.id == card_id)
            .values(
                stability=new_stability,
                difficulty=new_difficulty,
                review_count=SpacedRepetitionCard.review_count + 1,
                next_review_at=new_next_review
            )
        )
        await async_session.commit()

        async_session.expire_all()

        result = await async_session.execute(
            select(SpacedRepetitionCard).where(SpacedRepetitionCard.id == card_id)
        )
        updated_card = result.scalar_one()

        assert updated_card.stability == new_stability
        assert updated_card.difficulty == new_difficulty
        assert updated_card.review_count == 1

    @pytest.mark.asyncio
    async def test_create_review_log(self, async_session: AsyncSession):
        """Test logging a review."""
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
            response_time_ms=2500
        )
        async_session.add(review)
        await async_session.commit()

        assert review.id is not None
        assert review.rating == 4

    @pytest.mark.asyncio
    async def test_get_due_cards(self, async_session: AsyncSession):
        """Test querying cards due for review."""
        from app.models.user import User
        from app.models.spaced_repetition import SpacedRepetitionCard, Concept

        user = User(**UserFactory.create())
        concept = Concept(name="Test", description="Test")
        async_session.add_all([user, concept])
        await async_session.commit()

        # Create cards with different due dates
        past_card = SpacedRepetitionCard(
            user_id=user.id,
            concept_id=concept.id,
            difficulty=5.0,
            stability=2.5,
            retrievability=0.9,
            review_count=1,
            next_review_at=datetime.utcnow() - timedelta(days=1)  # Overdue
        )
        future_card = SpacedRepetitionCard(
            user_id=user.id,
            concept_id=concept.id,
            difficulty=5.0,
            stability=2.5,
            retrievability=0.9,
            review_count=1,
            next_review_at=datetime.utcnow() + timedelta(days=7)  # Not due
        )
        async_session.add_all([past_card, future_card])
        await async_session.commit()

        # Query due cards
        result = await async_session.execute(
            select(SpacedRepetitionCard)
            .where(SpacedRepetitionCard.next_review_at <= datetime.utcnow())
        )
        due_cards = result.scalars().all()

        assert len(due_cards) == 1


class TestAchievementCRUD:
    """Test Achievement entity CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_achievement(self, async_session: AsyncSession):
        """Test unlocking an achievement."""
        from app.models.user import User
        from app.models.gamification import UserAchievement

        user = User(**UserFactory.create())
        async_session.add(user)
        await async_session.commit()

        achievement = UserAchievement(
            user_id=user.id,
            achievement_type="STREAK_MILESTONE",
            name="First Week Streak",
            description="Maintained a 7-day streak",
            xp_reward=100
        )
        async_session.add(achievement)
        await async_session.commit()

        assert achievement.id is not None
        assert achievement.xp_reward == 100

    @pytest.mark.asyncio
    async def test_get_user_achievements(self, async_session: AsyncSession):
        """Test retrieving all achievements for a user."""
        from app.models.user import User
        from app.models.gamification import UserAchievement

        user = User(**UserFactory.create())
        async_session.add(user)
        await async_session.commit()

        # Create multiple achievements
        achievements = [
            UserAchievement(
                user_id=user.id,
                achievement_type="STREAK_MILESTONE",
                name="7-Day Streak",
                description="7-day streak",
                xp_reward=100
            ),
            UserAchievement(
                user_id=user.id,
                achievement_type="XP_MILESTONE",
                name="1000 XP",
                description="Earned 1000 XP",
                xp_reward=50
            ),
        ]
        async_session.add_all(achievements)
        await async_session.commit()

        # Query achievements
        result = await async_session.execute(
            select(UserAchievement).where(UserAchievement.user_id == user.id)
        )
        user_achievements = result.scalars().all()

        assert len(user_achievements) == 2
        total_xp = sum(a.xp_reward for a in user_achievements)
        assert total_xp == 150


class TestBulkOperations:
    """Test bulk database operations."""

    @pytest.mark.asyncio
    async def test_bulk_insert_users(self, async_session: AsyncSession):
        """Test bulk inserting multiple users."""
        from app.models.user import User

        users = [User(**UserFactory.create()) for _ in range(100)]
        async_session.add_all(users)
        await async_session.commit()

        result = await async_session.execute(select(User))
        all_users = result.scalars().all()

        assert len(all_users) == 100

    @pytest.mark.asyncio
    async def test_bulk_update(self, async_session: AsyncSession):
        """Test bulk updating multiple records."""
        from app.models.user import User

        # Create users with level 1
        users = [User(**UserFactory.create(level=1)) for _ in range(50)]
        async_session.add_all(users)
        await async_session.commit()

        # Bulk update all to level 2
        await async_session.execute(
            update(User)
            .where(User.level == 1)
            .values(level=2)
        )
        await async_session.commit()

        async_session.expire_all()

        result = await async_session.execute(
            select(User).where(User.level == 2)
        )
        updated_users = result.scalars().all()

        assert len(updated_users) == 50

    @pytest.mark.asyncio
    async def test_bulk_delete(self, async_session: AsyncSession):
        """Test bulk deleting multiple records."""
        from app.models.user import User

        # Create active and inactive users
        active_users = [User(**UserFactory.create(is_active=True)) for _ in range(30)]
        inactive_users = [User(**UserFactory.create(is_active=False)) for _ in range(20)]
        async_session.add_all(active_users + inactive_users)
        await async_session.commit()

        # Delete inactive users
        await async_session.execute(
            delete(User).where(User.is_active == False)  # noqa: E712
        )
        await async_session.commit()

        result = await async_session.execute(select(User))
        remaining_users = result.scalars().all()

        assert len(remaining_users) == 30
        assert all(u.is_active for u in remaining_users)
