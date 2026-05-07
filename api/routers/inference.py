import time

from celery.result import AsyncResult
from fastapi import APIRouter, Form, File, HTTPException, UploadFile, status

from api.celery_app import celery_app
from api.schemas.inference import InferenceEnqueueResponse, InferenceResponse
from api.services.gpu_lock import is_gpu_busy_with_training
from api.services.storage import save_upload_to_shared_dir
from api.tasks.inference import run_inference_task, normalize_model_key

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


def _enqueue_inference(file_bytes: bytes, filename: str, model_key: str | None) -> tuple[str, str]:
    upload_path = save_upload_to_shared_dir(
        filename=filename or "upload.bin",
        content=file_bytes,
        subdir="api_uploads",
    )

    queue = "infer_cpu" if is_gpu_busy_with_training() else "infer_gpu"

    task = run_inference_task.apply_async(
        kwargs={
            "payload": {
                "upload_path": upload_path,
                "source": "api",
                "requested_model_key": model_key,
                "requested_queue": queue,
                "enqueued_at_ts": time.time(),
            }
        },
        queue=queue,
    )

    return task.id, queue

@router.post(
    "/default/async",
    response_model=InferenceEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def inference_default_async(
        file: UploadFile = File(...),
        model_key: str | None = Form(default=None),
) -> InferenceEnqueueResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    normalized_model_key = normalize_model_key(model_key)

    task_id, queue = _enqueue_inference(
        file_bytes=file_bytes,
        filename=file.filename or "upload.bin",
        model_key=normalized_model_key,
    )

    return InferenceEnqueueResponse(
        task_id=task_id,
        status="queued",
        queue=queue,
    )


@router.post(
    "/default",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
)
async def inference_default(
    file: UploadFile = File(...),
    model_key: str | None = Form(default=None),
) -> InferenceResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    normalized_model_key = normalize_model_key(model_key)

    task_id, _queue = _enqueue_inference(
        file_bytes=file_bytes,
        filename=file.filename or "upload.bin",
        model_key=normalized_model_key
    )

    result = AsyncResult(task_id, app=celery_app)

    try:
        payload = result.get(timeout=180)
    except Exception as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Inference task did not finish in time: {exc}",
        )

    return InferenceResponse(
        predicted_class=payload["predicted_class"],
        confidence=payload["confidence"],
        probabilities=payload["probabilities"],
    )