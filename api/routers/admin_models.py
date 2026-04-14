from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.core.security import get_current_admin
from api.db.session import get_db
from api.repositories.model_registry import ModelRegistryRepository
from api.schemas.model_registry import (
    DefaultModelUpdate,
    RegisteredModelCreate,
    RegisteredModelRead,
)

router = APIRouter(
    prefix="/api/v1/admin/models",
    tags=["admin-models"],
    dependencies=[Depends(get_current_admin)],
)


@router.post("", response_model=RegisteredModelRead, status_code=status.HTTP_201_CREATED)
def create_model(payload: RegisteredModelCreate, db: Session = Depends(get_db)) -> RegisteredModelRead:
    repo = ModelRegistryRepository(db)
    existing = repo.get_by_model_key(payload.model_key)
    if existing is not None:
        raise HTTPException(status_code=409, detail="model_key already exists")
    model = repo.create(**payload.model_dump())
    return RegisteredModelRead.model_validate(model)


@router.get("", response_model=list[RegisteredModelRead])
def list_models(db: Session = Depends(get_db)) -> list[RegisteredModelRead]:
    repo = ModelRegistryRepository(db)
    return [RegisteredModelRead.model_validate(x) for x in repo.list_all()]


@router.get("/default", response_model=RegisteredModelRead)
def get_default_model(db: Session = Depends(get_db)) -> RegisteredModelRead:
    repo = ModelRegistryRepository(db)
    model = repo.get_default()
    if model is None:
        raise HTTPException(status_code=404, detail="Default model is not set")
    return RegisteredModelRead.model_validate(model)


@router.post("/default", response_model=RegisteredModelRead)
def set_default_model(payload: DefaultModelUpdate, db: Session = Depends(get_db)) -> RegisteredModelRead:
    repo = ModelRegistryRepository(db)
    try:
        model = repo.set_default(payload.model_key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RegisteredModelRead.model_validate(model)