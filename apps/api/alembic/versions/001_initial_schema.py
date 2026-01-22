"""Initial schema with all models

Revision ID: 001
Revises:
Create Date: 2026-01-08

This migration creates all tables for the NerdLearn platform including:
- User management (users, instructors)
- Course management (courses, modules, enrollments)
- Adaptive learning (spaced repetition, concepts, review logs)
- Assessment (user concept mastery)
- Gamification (achievements, stats, daily activities, chat history)

Run this migration after deploying to production:
    alembic upgrade head
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # =========================================================================
    # User Management Tables
    # =========================================================================

    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_instructor', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_username', 'users', ['username'])

    # Instructors table
    op.create_table(
        'instructors',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('expertise', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('rating', sa.Float(), default=0.0),
        sa.Column('total_students', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_instructors_user_id', 'instructors', ['user_id'])

    # =========================================================================
    # Course Management Tables
    # =========================================================================

    # Courses table
    op.create_table(
        'courses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('instructor_id', sa.Integer(), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('difficulty_level', sa.String(length=50), nullable=True),
        sa.Column('estimated_hours', sa.Integer(), nullable=True),
        sa.Column('is_published', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['instructor_id'], ['instructors.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_courses_instructor_id', 'courses', ['instructor_id'])
    op.create_index('ix_courses_category', 'courses', ['category'])

    # Modules table
    op.create_table(
        'modules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('order', sa.Integer(), nullable=False),
        sa.Column('content_type', sa.String(length=50), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('video_url', sa.String(length=500), nullable=True),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('processing_status', sa.String(length=50), default='pending'),
        sa.Column('processing_task_id', sa.String(length=255), nullable=True),
        sa.Column('chunk_count', sa.Integer(), default=0),
        sa.Column('concept_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_modules_course_id', 'modules', ['course_id'])
    op.create_index('ix_modules_processing_status', 'modules', ['processing_status'])

    # Enrollments table
    op.create_table(
        'enrollments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('enrolled_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('progress', sa.Float(), default=0.0),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_enrollments_user_id', 'enrollments', ['user_id'])
    op.create_index('ix_enrollments_course_id', 'enrollments', ['course_id'])

    # =========================================================================
    # Adaptive Learning Tables
    # =========================================================================

    # Concepts table
    op.create_table(
        'concepts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('module_id', sa.Integer(), nullable=True),
        sa.Column('difficulty', sa.Float(), default=0.5),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['module_id'], ['modules.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_concepts_course_id', 'concepts', ['course_id'])
    op.create_index('ix_concepts_module_id', 'concepts', ['module_id'])

    # Spaced Repetition Cards table
    op.create_table(
        'spaced_repetition_cards',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('concept_id', sa.Integer(), nullable=False),
        sa.Column('stability', sa.Float(), default=0.0),
        sa.Column('difficulty', sa.Float(), default=5.0),
        sa.Column('elapsed_days', sa.Integer(), default=0),
        sa.Column('scheduled_days', sa.Integer(), default=0),
        sa.Column('reps', sa.Integer(), default=0),
        sa.Column('lapses', sa.Integer(), default=0),
        sa.Column('state', sa.String(length=50), default='new'),
        sa.Column('last_review', sa.DateTime(timezone=True), nullable=True),
        sa.Column('due', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['concept_id'], ['concepts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_spaced_repetition_cards_user_id', 'spaced_repetition_cards', ['user_id'])
    op.create_index('ix_spaced_repetition_cards_concept_id', 'spaced_repetition_cards', ['concept_id'])
    op.create_index('ix_spaced_repetition_cards_due', 'spaced_repetition_cards', ['due'])
    op.create_index('ix_spaced_repetition_cards_state', 'spaced_repetition_cards', ['state'])

    # Review Logs table
    op.create_table(
        'review_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('card_id', sa.Integer(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('state', sa.String(length=50), nullable=False),
        sa.Column('due', sa.DateTime(timezone=True), nullable=False),
        sa.Column('stability', sa.Float(), nullable=False),
        sa.Column('difficulty', sa.Float(), nullable=False),
        sa.Column('elapsed_days', sa.Integer(), nullable=False),
        sa.Column('scheduled_days', sa.Integer(), nullable=False),
        sa.Column('review_duration_seconds', sa.Integer(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['card_id'], ['spaced_repetition_cards.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_review_logs_card_id', 'review_logs', ['card_id'])
    op.create_index('ix_review_logs_reviewed_at', 'review_logs', ['reviewed_at'])

    # =========================================================================
    # Assessment Tables
    # =========================================================================

    # User Concept Mastery table
    op.create_table(
        'user_concept_mastery',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('concept_id', sa.Integer(), nullable=False),
        sa.Column('mastery_level', sa.Float(), default=0.0),
        sa.Column('p_learned', sa.Float(), default=0.1),
        sa.Column('p_transit', sa.Float(), default=0.15),
        sa.Column('p_guess', sa.Float(), default=0.2),
        sa.Column('p_slip', sa.Float(), default=0.1),
        sa.Column('confidence', sa.Float(), default=0.0),
        sa.Column('last_assessed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('assessment_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['concept_id'], ['concepts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'concept_id', name='uq_user_concept')
    )
    op.create_index('ix_user_concept_mastery_user_id', 'user_concept_mastery', ['user_id'])
    op.create_index('ix_user_concept_mastery_concept_id', 'user_concept_mastery', ['concept_id'])

    # =========================================================================
    # Gamification Tables
    # =========================================================================

    # User Achievements table
    op.create_table(
        'user_achievements',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('achievement_id', sa.String(length=100), nullable=False),
        sa.Column('unlocked_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('xp_reward', sa.Integer(), default=0),
        sa.Column('rarity', sa.String(length=50), default='common'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_user_achievements_user_id', 'user_achievements', ['user_id'])
    op.create_index('ix_user_achievements_achievement_id', 'user_achievements', ['achievement_id'])

    # User Stats table
    op.create_table(
        'user_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('total_xp', sa.Integer(), default=0),
        sa.Column('current_level', sa.Integer(), default=1),
        sa.Column('current_streak_days', sa.Integer(), default=0),
        sa.Column('longest_streak_days', sa.Integer(), default=0),
        sa.Column('last_activity_date', sa.Date(), nullable=True),
        sa.Column('modules_completed', sa.Integer(), default=0),
        sa.Column('concepts_mastered', sa.Integer(), default=0),
        sa.Column('reviews_completed', sa.Integer(), default=0),
        sa.Column('chat_interactions', sa.Integer(), default=0),
        sa.Column('achievements_unlocked', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index('ix_user_stats_user_id', 'user_stats', ['user_id'])
    op.create_index('ix_user_stats_total_xp', 'user_stats', ['total_xp'])

    # Daily Activities table
    op.create_table(
        'daily_activities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('xp_earned', sa.Integer(), default=0),
        sa.Column('modules_completed', sa.Integer(), default=0),
        sa.Column('reviews_completed', sa.Integer(), default=0),
        sa.Column('chat_interactions', sa.Integer(), default=0),
        sa.Column('time_spent_minutes', sa.Integer(), default=0),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'date', name='uq_user_date')
    )
    op.create_index('ix_daily_activities_user_id', 'daily_activities', ['user_id'])
    op.create_index('ix_daily_activities_date', 'daily_activities', ['date'])

    # Chat History table
    op.create_table(
        'chat_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('module_id', sa.Integer(), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('response', sa.Text(), nullable=False),
        sa.Column('citations', postgresql.JSONB(), nullable=True),
        sa.Column('xp_awarded', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['module_id'], ['modules.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_chat_history_user_id', 'chat_history', ['user_id'])
    op.create_index('ix_chat_history_course_id', 'chat_history', ['course_id'])
    op.create_index('ix_chat_history_created_at', 'chat_history', ['created_at'])


def downgrade() -> None:
    # Drop tables in reverse order to respect foreign key constraints
    op.drop_table('chat_history')
    op.drop_table('daily_activities')
    op.drop_table('user_stats')
    op.drop_table('user_achievements')
    op.drop_table('user_concept_mastery')
    op.drop_table('review_logs')
    op.drop_table('spaced_repetition_cards')
    op.drop_table('concepts')
    op.drop_table('enrollments')
    op.drop_table('modules')
    op.drop_table('courses')
    op.drop_table('instructors')
    op.drop_table('users')
