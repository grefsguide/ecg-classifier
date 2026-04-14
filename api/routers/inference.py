from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.db.session import get_db
from api.repositories.model_registry import ModelRegistryRepository
from api.schemas.inference import InferenceResponse
from api.services.inference import DEFAULT_CLASS_NAMES, run_inference

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

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = run_inference(
            file_bytes=file_bytes,
            checkpoint_path=model.checkpoint_path,
            model_name=model.model_name,
            class_names=list(model.config_snapshot.get("class_names", DEFAULT_CLASS_NAMES)),
            config_snapshot=model.config_snapshot,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load checkpoint or run inference: {exc}",
        ) from exc

    return InferenceResponse(
        model_key=model.model_key,
        model_name=model.model_name,
        checkpoint_path=model.checkpoint_path,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )