"""add_completion_tracking

Revision ID: 3f9a2c1d8b6
Revises: <previous_revision_id>
Create Date: 2026-03-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers
revision = "3f9a2c1d8b6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "roadmaps",
        sa.Column("completed_problems", JSON, nullable=True)
    )

    op.add_column(
        "roadmaps",
        sa.Column("progress_percentage", sa.Float, server_default="0.0")
    )

    op.add_column(
        "roadmaps",
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now())
    )


def downgrade():
    op.drop_column("roadmaps", "updated_at")
    op.drop_column("roadmaps", "progress_percentage")
    op.drop_column("roadmaps", "completed_problems")