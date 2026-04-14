from typing import Any

from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    model_name: str
    split_name: str = "default"
    max_epochs: int = 1
    batch_size: int | None = None
    img_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    pretrained: bool | None = None
    timm_name: str | None = None
    make_default: bool = False
    display_name: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    extra_overrides: list[str] = Field(default_factory=list)


class ExperimentTaskResponse(BaseModel):
    task_id: str
    status: str