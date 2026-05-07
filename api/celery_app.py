import os

from celery import Celery
from kombu import Queue

from api.core.settings import settings
from api.observability.metrics import start_worker_metrics_server

VISIBILITY_TIMEOUT = 120 * 60 * 60

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
    broker_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT},
    result_backend_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT},
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

if settings.prometheus_enabled and os.getenv("PROMETHEUS_WORKER_SERVER_ENABLED", "false").lower() == "true":
    start_worker_metrics_server(settings.prometheus_worker_metrics_port)