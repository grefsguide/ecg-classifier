from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RegisteredModelCreate(BaseModel):
    model_key: str
    display_name: str
    model_name: str
    checkpoint_path: str
    split_name: str | None = None
    mlflow_run_id: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)


class RegisteredModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_key: str
    display_name: str
    model_name: str
    checkpoint_path: str
    split_name: str | None
    mlflow_run_id: str | None
    config_snapshot: dict[str, Any]
    metrics: dict[str, Any]
    tags: dict[str, Any]
    is_active: bool
    is_default: bool
    created_at: datetime | None
    updated_at: datetime | None


class DefaultModelUpdate(BaseModel):
    model_key: str