from pydantic import BaseModel


class InferenceResponse(BaseModel):
    model_key: str
    model_name: str
    checkpoint_path: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]