from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TelegramInferenceEnqueueResponse(BaseModel):
    task_id: str
    status: str
    queue: str
    model_key: str
    model_name: str


class TelegramInferenceHistoryItem(BaseModel):
    task_id: str
    status: str
    original_filename: str | None = None
    predicted_class: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    error_message: str | None = None
    created_at: datetime
    image_url: str | None = None


class TelegramHistoryResponse(BaseModel):
    items: list[TelegramInferenceHistoryItem] = Field(default_factory=list)