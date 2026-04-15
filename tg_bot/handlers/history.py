import logging

import httpx
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message

from tg_bot.services.api_client import ApiClient

router = Router()
api_client = ApiClient()
logger = logging.getLogger(__name__)


@router.message(Command("history"))
async def history_command(message: Message) -> None:
    try:
        payload = await api_client.get_history(message.from_user.id, limit=10)
        items = payload.get("items", [])

        if not items:
            await message.answer("История пока пуста.")
            return

        await message.answer("Последние результаты:")

        async with httpx.AsyncClient(timeout=60.0) as client:
            for item in items:
                caption = _build_caption(item)
                image_url = item.get("image_url")

                if image_url:
                    try:
                        response = await client.get(image_url)
                        response.raise_for_status()

                        filename = item.get("original_filename") or f"{item['task_id']}.jpg"
                        photo = BufferedInputFile(
                            file=response.content,
                            filename=filename,
                        )

                        await message.answer_photo(
                            photo=photo,
                            caption=caption,
                        )
                        continue
                    except Exception:
                        logger.exception(
                            "Failed to load history image for task_id=%s",
                            item.get("task_id"),
                        )

                await message.answer(caption)

    except Exception as exc:
        logger.exception("Failed to fetch telegram history")
        await message.answer(f"Не удалось получить историю: {exc}")


def _build_caption(item: dict) -> str:
    created_at = item.get("created_at", "—")
    predicted_class = item.get("predicted_class") or "—"
    status = item.get("status", "—")
    filename = item.get("original_filename") or "—"

    confidence = item.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_text = f"{confidence:.2f}"
    else:
        confidence_text = "—"

    error_message = item.get("error_message")

    lines = [
        f"Дата: {created_at}",
        f"Файл: {filename}",
        f"Статус: {status}",
        f"Результат: {predicted_class}",
        f"Уверенность: {confidence_text}",
    ]

    if error_message:
        lines.append(f"Ошибка: {error_message}")

    return "\n".join(lines)