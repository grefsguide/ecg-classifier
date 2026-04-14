from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.db.session import get_db
from api.repositories.model_registry import ModelRegistryRepository
from api.schemas.inference import InferenceResponse

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


@router.post("/default", response_model=InferenceResponse)
async def inference_default(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> InferenceResponse:
    repo = ModelRegistryRepository(db)
    model = repo.get_default()
    if model is None:
        raise HTTPException(status_code=404, detail="Default model is not set")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    return InferenceResponse(
        model_key=model.model_key,
        model_name=model.model_name,
        checkpoint_path=model.checkpoint_path,
        predicted_class="NORM",
        confidence=0.95,
    )