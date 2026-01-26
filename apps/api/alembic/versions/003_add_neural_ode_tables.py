"""add neural ode tables

Revision ID: 003
Revises: 002
Create Date: 2026-01-25 12:00:00.000000

Adds tables for CT-MCN (Continuous-Time Memory Calibration Network):
- memory_state_trajectory: h(t) latent state time series
- user_circadian_pattern: Per-user circadian parameters
- learner_phenotype: Learner type cluster assignments
- ode_card_state: Per user-concept ODE state
- response_time_observation: RT metrics for TD-BKT
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. memory_state_trajectory - Stores h(t) time series
    op.create_table(
        'memory_state_trajectory',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('concept_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('latent_state', postgresql.JSONB(astext_type=sa.Text()), nullable=False),  # 32-dim vector
        sa.Column('predicted_retrievability', sa.Float(), nullable=False),
        sa.Column('uncertainty_epistemic', sa.Float(), nullable=True),
        sa.Column('uncertainty_aleatoric', sa.Float(), nullable=True),
        sa.Column('circadian_factor', sa.Float(), nullable=True),
        sa.Column('sleep_consolidated', sa.Boolean(), server_default='false'),
        sa.Column('stress_coefficient', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_memory_state_trajectory_user_concept', 'memory_state_trajectory', ['user_id', 'concept_id'])
    op.create_index('ix_memory_state_trajectory_timestamp', 'memory_state_trajectory', ['timestamp'])

    # 2. user_circadian_pattern - Per-user circadian parameters
    op.create_table(
        'user_circadian_pattern',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('peak_hours', postgresql.JSONB(astext_type=sa.Text()), nullable=False),  # e.g., [9, 10, 14, 15]
        sa.Column('amplitude', sa.Float(), server_default='0.2'),
        sa.Column('phase_offset', sa.Float(), server_default='0.0'),
        sa.Column('typical_sleep_start', sa.Integer(), server_default='23'),  # hour 0-23
        sa.Column('typical_sleep_end', sa.Integer(), server_default='7'),  # hour 0-23
        sa.Column('chronotype', sa.String(20), server_default='neutral'),  # morning_lark, night_owl, neutral
        sa.Column('last_detected_sleep_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_detected_sleep_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', name='uq_user_circadian_pattern_user_id')
    )

    # 3. learner_phenotype - Cluster assignments
    op.create_table(
        'learner_phenotype',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('phenotype_id', sa.Integer(), nullable=False),  # cluster index 0-6
        sa.Column('phenotype_name', sa.String(50), nullable=False),  # "Fast Forgetter", etc.
        sa.Column('centroid_params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('assignment_confidence', sa.Float(), server_default='0.5'),
        sa.Column('reviews_at_assignment', sa.Integer(), server_default='0'),
        sa.Column('decay_rate_factor', sa.Float(), server_default='1.0'),  # relative to population
        sa.Column('learning_rate_factor', sa.Float(), server_default='1.0'),  # relative to population
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', name='uq_learner_phenotype_user_id')
    )
    op.create_index('ix_learner_phenotype_phenotype_id', 'learner_phenotype', ['phenotype_id'])

    # 4. ode_card_state - Per user-concept ODE state
    op.create_table(
        'ode_card_state',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('concept_id', sa.Integer(), nullable=False),
        sa.Column('current_latent_state', postgresql.JSONB(astext_type=sa.Text()), nullable=False),  # 32-dim vector
        sa.Column('last_state_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('control_mode', sa.String(20), server_default='shadow'),  # shadow, hybrid, active
        sa.Column('ode_confidence', sa.Float(), server_default='0.0'),
        sa.Column('review_count', sa.Integer(), server_default='0'),
        sa.Column('total_training_samples', sa.Integer(), server_default='0'),
        sa.Column('last_loss', sa.Float(), nullable=True),  # last training loss
        sa.Column('model_version', sa.Integer(), server_default='0'),  # increments on retrain
        sa.Column('min_interval_days', sa.Integer(), server_default='1'),
        sa.Column('max_interval_days', sa.Integer(), server_default='365'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'concept_id', name='uq_ode_card_state_user_concept')
    )
    op.create_index('ix_ode_card_state_user_id', 'ode_card_state', ['user_id'])
    op.create_index('ix_ode_card_state_control_mode', 'ode_card_state', ['control_mode'])

    # 5. response_time_observation - RT metrics for TD-BKT
    op.create_table(
        'response_time_observation',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('review_log_id', sa.Integer(), nullable=False),
        sa.Column('response_time_ms', sa.Integer(), nullable=False),
        sa.Column('normalized_rt', sa.Float(), nullable=True),  # log-normalized
        sa.Column('cursor_tortuosity', sa.Float(), nullable=True),  # path_length / euclidean_distance
        sa.Column('hesitation_count', sa.Integer(), nullable=True),  # pauses > 500ms
        sa.Column('backspace_count', sa.Integer(), nullable=True),
        sa.Column('click_count', sa.Integer(), nullable=True),
        sa.Column('retrieval_fluency', sa.Float(), nullable=True),  # computed metric 0-1
        sa.Column('time_to_first_action_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['review_log_id'], ['review_logs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_response_time_observation_review_log_id', 'response_time_observation', ['review_log_id'])

    # 6. Add ODE columns to spaced_repetition_cards
    op.add_column('spaced_repetition_cards', sa.Column('ode_enabled', sa.Boolean(), server_default='false'))
    op.add_column('spaced_repetition_cards', sa.Column('ode_predicted_interval', sa.Integer(), nullable=True))
    op.add_column('spaced_repetition_cards', sa.Column('ode_confidence', sa.Float(), nullable=True))

    # 7. Add RT and context columns to review_logs
    op.add_column('review_logs', sa.Column('response_time_ms', sa.Integer(), nullable=True))
    op.add_column('review_logs', sa.Column('contextual_factors', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    # Remove columns from review_logs
    op.drop_column('review_logs', 'contextual_factors')
    op.drop_column('review_logs', 'response_time_ms')

    # Remove columns from spaced_repetition_cards
    op.drop_column('spaced_repetition_cards', 'ode_confidence')
    op.drop_column('spaced_repetition_cards', 'ode_predicted_interval')
    op.drop_column('spaced_repetition_cards', 'ode_enabled')

    # Drop tables in reverse order
    op.drop_index('ix_response_time_observation_review_log_id', table_name='response_time_observation')
    op.drop_table('response_time_observation')

    op.drop_index('ix_ode_card_state_control_mode', table_name='ode_card_state')
    op.drop_index('ix_ode_card_state_user_id', table_name='ode_card_state')
    op.drop_table('ode_card_state')

    op.drop_index('ix_learner_phenotype_phenotype_id', table_name='learner_phenotype')
    op.drop_table('learner_phenotype')

    op.drop_table('user_circadian_pattern')

    op.drop_index('ix_memory_state_trajectory_timestamp', table_name='memory_state_trajectory')
    op.drop_index('ix_memory_state_trajectory_user_concept', table_name='memory_state_trajectory')
    op.drop_table('memory_state_trajectory')
