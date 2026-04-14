from sqlalchemy import select, update
from sqlalchemy.orm import Session

from api.models.model_registry import RegisteredModel


class ModelRegistryRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> RegisteredModel:
        obj = RegisteredModel(**kwargs)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def list_all(self) -> list[RegisteredModel]:
        stmt = select(RegisteredModel).order_by(RegisteredModel.created_at.desc())
        return list(self.db.scalars(stmt).all())

    def get_by_model_key(self, model_key: str) -> RegisteredModel | None:
        stmt = select(RegisteredModel).where(RegisteredModel.model_key == model_key)
        return self.db.scalar(stmt)

    def get_default(self) -> RegisteredModel | None:
        stmt = select(RegisteredModel).where(RegisteredModel.is_default.is_(True))
        return self.db.scalar(stmt)

    def set_default(self, model_key: str) -> RegisteredModel:
        target = self.get_by_model_key(model_key)
        if target is None:
            raise ValueError(f"Model '{model_key}' not found")

        self.db.execute(update(RegisteredModel).values(is_default=False))
        self.db.execute(
            update(RegisteredModel)
            .where(RegisteredModel.model_key == model_key)
            .values(is_default=True)
        )
        self.db.commit()
        self.db.refresh(target)
        return target
