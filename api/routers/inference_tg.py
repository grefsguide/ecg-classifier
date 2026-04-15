from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from api.db.session import get_db
from api.repositories.tg_history import TelegramHistoryRepository
from api.schemas.tg import TelegramInferenceEnqueueResponse
from api.services.gpu_lock import is_gpu_busy_with_training
from api.services.storage import save_upload_to_shared_dir
from api.tasks.inference import run_inference_task

router = APIRouter(prefix="/api/v1/inference-tg", tags=["inference_tg"])


@router.post(
    "/default",
    response_model=TelegramInferenceEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def inference_tg_default(
    file: UploadFile = File(...),
    telegram_user_id: int = Form(...),
    telegram_username: str | None = Form(default=None),
    telegram_display_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> TelegramInferenceEnqueueResponse:
    history_repo = TelegramHistoryRepository(db)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    upload_path = save_upload_to_shared_dir(
        filename=file.filename or "upload.bin",
        content=file_bytes,
        subdir="telegram_uploads",
    )

    history_repo.upsert_user(
        telegram_user_id=telegram_user_id,
        username=telegram_username,
        display_name=telegram_display_name,
    )

    queue = "infer_cpu" if is_gpu_busy_with_training() else "infer_gpu"

    task = run_inference_task.apply_async(
        kwargs={
            "payload": {
                "upload_path": upload_path,
                "source": "telegram",
                "telegram_user_id": telegram_user_id,
            }
        },
        queue=queue,
    )

    history_repo.create_history(
        task_id=task.id,
        telegram_user_id=telegram_user_id,
        image_path=upload_path,
        original_filename=file.filename,
        status="queued",
        queue_name=queue,
        model_key=None,
        model_name=None,
    )
    db.commit()

    return TelegramInferenceEnqueueResponse(
        task_id=task.id,
        status="queued",
        queue=queue,
        model_key="default",
        model_name="default",
    )