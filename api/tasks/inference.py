from api.celery_app import celery_app
from api.db.session import SessionLocal
from api.repositories.model_registry import ModelRegistryRepository
from api.repositories.tg_history import TelegramHistoryRepository
from api.services.inference import DEFAULT_CLASS_NAMES, run_inference
from api.services.storage import read_uploaded_file_bytes


@celery_app.task(name="api.tasks.inference.run_inference", bind=True)
def run_inference_task(self, payload: dict) -> dict:
    db = SessionLocal()

    try:
        repo = ModelRegistryRepository(db)
        model = repo.get_default()
        if model is None:
            raise ValueError("Default model is not set")

        file_bytes = read_uploaded_file_bytes(payload["upload_path"])

        result = run_inference(
            file_bytes=file_bytes,
            checkpoint_path=model.checkpoint_path,
            model_name=model.model_name,
            class_names=list(
                model.config_snapshot.get("class_names", DEFAULT_CLASS_NAMES)
            ),
            config_snapshot=model.config_snapshot,
        )

        if payload.get("source") == "telegram":
            history_repo = TelegramHistoryRepository(db)
            history_repo.mark_completed(
                task_id=self.request.id,
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                probabilities=result.probabilities,
            )
            db.commit()

        return {
            "task_id": self.request.id,
            "status": "completed",
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        }

    except Exception as exc:
        if payload.get("source") == "telegram":
            history_repo = TelegramHistoryRepository(db)
            history_repo.mark_failed(
                task_id=self.request.id,
                error_message=str(exc),
            )
            db.commit()
        raise
    finally:
        db.close()