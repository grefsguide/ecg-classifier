from celery import Celery

from api.core.settings import settings

celery_app = Celery(
    "ecg_api",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["api.tasks.experiments"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    imports=("api.tasks.experiments",),
)