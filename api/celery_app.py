from celery import Celery
from kombu import Queue

from api.core.settings import settings

celery_app = Celery(
    "ecg_api",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "api.tasks.experiments",
        "api.tasks.inference",
    ],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_queues=(
        Queue("train"),
        Queue("infer_gpu"),
        Queue("infer_cpu"),
    ),
    task_routes={
        "api.tasks.experiments.run_experiment_task": {"queue": "train"},
        "api.tasks.inference.run_inference": {"queue": "infer_gpu"},
    },
)