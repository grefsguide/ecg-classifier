from __future__ import annotations

import logging

from api.celery_app import celery_app
from api.db.session import SessionLocal
from api.repositories.model_registry import ModelRegistryRepository
from api.services.training import run_training_pipeline

logger = logging.getLogger(__name__)


@celery_app.task(name="run_experiment", bind=True)
def run_experiment_task(self, payload: dict) -> dict:
    logger.info("Experiment task started: %s", payload)

    db = SessionLocal()
    try:
        result = run_training_pipeline(payload)

        repo = ModelRegistryRepository(db)

        created_model = repo.create(
            model_key=result["model_key"],
            display_name=result["display_name"],
            model_name=result["model_name"],
            checkpoint_path=result["checkpoint_path"],
            split_name=result["split_name"],
            mlflow_run_id=result["mlflow_run_id"],
            config_snapshot=result["config_snapshot"],
            metrics=result["metrics"],
            tags=result["tags"],
        )

        if result["make_default"]:
            created_model = repo.set_default(created_model.model_key)

        response = {
            "task_id": self.request.id,
            "status": "completed",
            "model_key": created_model.model_key,
            "checkpoint_path": created_model.checkpoint_path,
            "mlflow_run_id": created_model.mlflow_run_id,
            "metrics": created_model.metrics,
            "is_default": created_model.is_default,
            "applied_overrides": result["applied_overrides"],
            "metrics_path": result["metrics_path"],
        }

        logger.info("Experiment task finished: %s", response)
        return response

    except Exception:
        logger.exception("Experiment task failed")
        raise
    finally:
        db.close()