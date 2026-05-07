from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RegisteredModelCreate(BaseModel):
    model_key: str
    display_name: str
    model_name: str
    checkpoint_path: str
    checkpoint_uri: str | None = None
    split_name: str | None = None
    mlflow_run_id: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metrics_uri: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    storage_backend: str = "local"


class RegisteredModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_key: str
    display_name: str
    model_name: str
    checkpoint_path: str
    checkpoint_uri: str | None = None
    split_name: str | None
    mlflow_run_id: str | None
    config_snapshot: dict[str, Any]
    metrics: dict[str, Any]
    metrics_uri: str | None = None
    tags: dict[str, Any]
    is_active: bool
    is_default: bool
    created_at: datetime | None
    updated_at: datetime | None
    storage_backend: str


class DefaultModelUpdate(BaseModel):
    model_key: str