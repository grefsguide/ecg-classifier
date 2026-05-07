from time import perf_counter, time

from api.celery_app import celery_app
from api.db.session import SessionLocal
from api.observability.metrics import inc_inference_requests_total, observe_inference_total, observe_queue_wait
from api.repositories.model_registry import ModelRegistryRepository
from api.repositories.tg_history import TelegramHistoryRepository
from api.services.inference import DEFAULT_CLASS_NAMES, get_device, run_inference
from api.services.storage import read_uploaded_file_bytes

def normalize_model_key(model_key: str | None) -> str | None:
    if model_key is None:
        return None

    model_key = model_key.strip()
    if not model_key:
        return None

    return model_key

@celery_app.task(name="api.tasks.inference.run_inference", bind=True)
def run_inference_task(self, payload: dict) -> dict:
    db = SessionLocal()
    task_started = perf_counter()

    source = str(payload.get("source", "api"))
    queue = str(payload.get("requested_queue", "unknown"))
    requested_model_key = normalize_model_key(payload.get("requested_model_key"))
    enqueued_at_ts = float(payload.get("enqueued_at_ts", time()))
    device = get_device().type

    try:
        repo = ModelRegistryRepository(db)

        if requested_model_key:
            model = repo.get_by_model_key(requested_model_key)
            if model is None:
                raise ValueError(f"Model '{requested_model_key}' not found")
        else:
            model = repo.get_default()

        if model is None:
            raise ValueError("Default model is not set")

        observe_queue_wait(
            queue=queue,
            source=source,
            seconds=max(0.0, time() - enqueued_at_ts),
        )

        file_bytes = read_uploaded_file_bytes(payload["upload_path"])

        config_snapshot = dict(model.config_snapshot or {})
        tags = dict(model.tags or {})

        if model.model_name == "resnet":
            if not config_snapshot.get("backbone_name"):
                config_snapshot["backbone_name"] = tags.get("backbone", "resnet18")

            if "pretrained" not in config_snapshot:
                config_snapshot["pretrained"] = tags.get("pretrained") in {"true", "imagenet", "1", True}

        result = run_inference(
            file_bytes=file_bytes,
            checkpoint_path=model.checkpoint_path,
            model_name=model.model_name,
            model_key=model.model_key,
            class_names=list(model.config_snapshot.get("class_names", DEFAULT_CLASS_NAMES)),
            config_snapshot=model.config_snapshot,
            source=source,
        )

        if source == "telegram":
            history_repo = TelegramHistoryRepository(db)
            history_repo.mark_completed(
                task_id=self.request.id,
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                probabilities=result.probabilities,
            )
            db.commit()

        inc_inference_requests_total(
            model_name=model.model_name,
            model_key=model.model_key,
            queue=queue,
            device=device,
            source=source,
            status="success",
        )

        observe_inference_total(
            model_name=model.model_name,
            model_key=model.model_key,
            queue=queue,
            device=device,
            source=source,
            status="success",
            seconds=perf_counter() - task_started,
        )

        return {
            "task_id": self.request.id,
            "status": "completed",
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        }

    except Exception as exc:
        model_name = "unknown"
        model_key = requested_model_key or "default"

        inc_inference_requests_total(
            model_name=model_name,
            model_key=model_key,
            queue=queue,
            device=device,
            source=source,
            status="error",
        )

        observe_inference_total(
            model_name=model_name,
            model_key=model_key,
            queue=queue,
            device=device,
            source=source,
            status="error",
            seconds=perf_counter() - task_started,
        )

        if source == "telegram":
            history_repo = TelegramHistoryRepository(db)
            history_repo.mark_failed(
                task_id=self.request.id,
                error_message=str(exc),
            )
            db.commit()

        raise
    finally:
        db.close()