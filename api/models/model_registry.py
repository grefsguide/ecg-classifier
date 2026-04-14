from datetime import datetime
from uuid import uuid4, UUID

from sqlalchemy import Boolean, DateTime, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from api.db.base import Base


class RegisteredModel(Base):
    __tablename__ = "registered_models"
    __table_args__ = (
        Index(
            "uq_registered_models_single_default",
            "is_default",
            unique=True,
            postgresql_where=text("is_default = true"),
        ),
    )

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    model_name: Mapped[str] = mapped_column(String(64))
    checkpoint_path: Mapped[str] = mapped_column(Text)
    split_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    tags: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=datetime.utcnow
    )
