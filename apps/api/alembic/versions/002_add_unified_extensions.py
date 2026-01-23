"""add unified extensions

Revision ID: 002
Revises: 001
Create Date: 2024-01-22 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS age")
    op.execute("LOAD 'age'") 
    op.execute("SET search_path = ag_catalog, \"$user\", public")

    # Table
    op.create_table('course_chunks',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=True),
        sa.Column('module_id', sa.Integer(), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('module_type', sa.String(), nullable=True),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('heading', sa.String(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('timestamp_start', sa.Float(), nullable=True),
        sa.Column('timestamp_end', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
        sa.ForeignKeyConstraint(['module_id'], ['modules.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_course_chunks_course_id'), 'course_chunks', ['course_id'], unique=False)
    op.create_index(op.f('ix_course_chunks_module_id'), 'course_chunks', ['module_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_course_chunks_module_id'), table_name='course_chunks')
    op.drop_index(op.f('ix_course_chunks_course_id'), table_name='course_chunks')
    op.drop_table('course_chunks')
    op.execute("DROP EXTENSION IF NOT EXISTS age")
    op.execute("DROP EXTENSION IF NOT EXISTS vector")
