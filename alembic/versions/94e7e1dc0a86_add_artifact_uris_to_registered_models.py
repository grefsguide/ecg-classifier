"""add artifact uris to registered models

Revision ID: 94e7e1dc0a86
Revises: 2426b22f452a
Create Date: 2026-05-07 04:58:54.137521

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



# revision identifiers, used by Alembic.
revision: str = '94e7e1dc0a86'
down_revision: Union[str, None] = '2426b22f452a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "registered_models",
        sa.Column("checkpoint_uri", sa.Text(), nullable=True),
    )
    op.add_column(
        "registered_models",
        sa.Column("metrics_uri", sa.Text(), nullable=True),
    )
    op.add_column(
        "registered_models",
        sa.Column(
            "storage_backend",
            sa.String(length=32),
            server_default="local",
            nullable=False,
        ),
    )

    op.execute(
        """
        UPDATE registered_models
        SET checkpoint_uri = checkpoint_path
        WHERE checkpoint_uri IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("registered_models", "storage_backend")
    op.drop_column("registered_models", "metrics_uri")
    op.drop_column("registered_models", "checkpoint_uri")