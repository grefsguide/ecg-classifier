from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.core.settings import settings
from api.db.session import get_db
from api.repositories.tg_history import TelegramHistoryRepository
from api.schemas.tg import TelegramHistoryResponse, TelegramInferenceHistoryItem

router = APIRouter(prefix="/api/v1/telegram", tags=["telegram"])


@router.get("/history/{telegram_user_id}", response_model=TelegramHistoryResponse)
def get_telegram_history(
    telegram_user_id: int,
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> TelegramHistoryResponse:
    repo = TelegramHistoryRepository(db)
    items = repo.list_history(telegram_user_id=telegram_user_id, limit=limit)

    base_url = settings.internal_api_base_url.rstrip("/")

    response_items: list[TelegramInferenceHistoryItem] = []

    for item in items:
        image_url = f"{base_url}/api/v1/telegram/history/image/{item.task_id}"

        response_items.append(
            TelegramInferenceHistoryItem(
                task_id=item.task_id,
                status=item.status,
                original_filename=item.original_filename,
                predicted_class=item.predicted_class,
                confidence=item.confidence,
                probabilities=item.probabilities,
                error_message=item.error_message,
                created_at=item.created_at,
                image_url=image_url,
            )
        )

    return TelegramHistoryResponse(items=response_items)


@router.get("/history/image/{task_id}", name="get_telegram_history_image")
def get_telegram_history_image(
    task_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    repo = TelegramHistoryRepository(db)
    item = repo.get_by_task_id(task_id)

    if item is None:
        raise HTTPException(status_code=404, detail="History item not found")

    file_path = Path(item.image_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    media_type = _guess_media_type(file_path)

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=item.original_filename or file_path.name,
    )


def _guess_media_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    if suffix == ".gif":
        return "image/gif"

    return "application/octet-stream"