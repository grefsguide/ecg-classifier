from pydantic import BaseModel


class InferenceEnqueueResponse(BaseModel):
    task_id: str
    status: str
    queue: str


class InferenceResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]