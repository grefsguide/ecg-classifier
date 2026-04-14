"""create registered_models

Revision ID: 20260414_000001
Revises:
Create Date: 2026-04-14 12:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260414_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "registered_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("model_key", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_path", sa.Text(), nullable=False),
        sa.Column("split_name", sa.String(length=128), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=128), nullable=True),
        sa.Column("config_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(op.f("ix_registered_models_is_default"), "registered_models", ["is_default"], unique=False)
    op.create_index(op.f("ix_registered_models_model_key"), "registered_models", ["model_key"], unique=True)
    op.create_index(
        "uq_registered_models_single_default",
        "registered_models",
        ["is_default"],
        unique=True,
        postgresql_where=sa.text("is_default = true"),
    )


def downgrade() -> None:
    op.drop_index("uq_registered_models_single_default", table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_model_key"), table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_is_default"), table_name="registered_models")
    op.drop_table("registered_models")
