from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.tg_history import TelegramInferenceHistory
from api.models.tg_user import TelegramUser


def resolve_telegram_login(username: str | None, telegram_user_id: int) -> str:
    if username:
        return f"@{username.lstrip('@')}"
    return f"tg_{telegram_user_id}"


class TelegramHistoryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_user(
        self,
        telegram_user_id: int,
        username: str | None,
        display_name: str | None,
    ) -> TelegramUser:
        user = self.db.get(TelegramUser, telegram_user_id)
        resolved_login = resolve_telegram_login(username, telegram_user_id)

        if user is None:
            user = TelegramUser(
                telegram_user_id=telegram_user_id,
                username=username,
                display_name=display_name,
                resolved_login=resolved_login,
            )
            self.db.add(user)
        else:
            user.username = username
            user.display_name = display_name
            user.resolved_login = resolved_login

        self.db.flush()
        return user

    def create_history(
        self,
        *,
        task_id: str,
        telegram_user_id: int,
        image_path: str,
        original_filename: str | None,
        status: str,
        queue_name: str | None,
        model_key: str | None,
        model_name: str | None,
    ) -> TelegramInferenceHistory:
        item = TelegramInferenceHistory(
            task_id=task_id,
            telegram_user_id=telegram_user_id,
            image_path=image_path,
            original_filename=original_filename,
            status=status,
            queue_name=queue_name,
            model_key=model_key,
            model_name=model_name,
        )
        self.db.add(item)
        self.db.flush()
        return item

    def get_by_task_id(self, task_id: str) -> TelegramInferenceHistory | None:
        stmt = select(TelegramInferenceHistory).where(
            TelegramInferenceHistory.task_id == task_id
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def mark_completed(
        self,
        *,
        task_id: str,
        predicted_class: str,
        confidence: float | None,
        probabilities: dict | None,
    ) -> None:
        item = self.get_by_task_id(task_id)
        if item is None:
            return

        item.status = "completed"
        item.predicted_class = predicted_class
        item.confidence = round(confidence, 2) if confidence is not None else None
        item.probabilities = probabilities

    def mark_failed(self, *, task_id: str, error_message: str) -> None:
        item = self.get_by_task_id(task_id)
        if item is None:
            return

        item.status = "failed"
        item.error_message = error_message

    def list_history(self, telegram_user_id: int, limit: int = 10) -> list[TelegramInferenceHistory]:
        stmt = (
            select(TelegramInferenceHistory)
            .where(TelegramInferenceHistory.telegram_user_id == telegram_user_id)
            .order_by(TelegramInferenceHistory.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())