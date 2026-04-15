import logging

from aiogram import F, Router
from aiogram.types import Message

from tg_bot.services.api_client import ApiClient
from tg_bot.services.tg_files import build_display_name, download_telegram_file

router = Router()
api_client = ApiClient()
logger = logging.getLogger(__name__)


@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    photo = message.photo[-1]
    tg_file = await message.bot.get_file(photo.file_id)

    local_path = await download_telegram_file(
        bot=message.bot,
        telegram_path=tg_file.file_path,
        filename=f"{photo.file_unique_id}.jpg",
    )

    processing_msg = await message.answer("Изображение получено, запускаю анализ...")

    try:
        enqueue = await api_client.submit_inference(
            file_path=local_path,
            filename=local_path.name,
            telegram_user_id=message.from_user.id,
            telegram_username=message.from_user.username,
            telegram_display_name=build_display_name(message),
        )

        task_id = enqueue.get("task_id")
        logger.info("Telegram inference task created: task_id=%s", task_id)

        if not task_id:
            await processing_msg.edit_text(
                f"Ошибка запуска анализа: API не вернул task_id.\nОтвет: {enqueue}"
            )
            return

        result = await api_client.wait_for_result(task_id)

        status_value = result.get("status")
        if status_value in {"FAILURE", "failed"}:
            await processing_msg.edit_text(
                f"Ошибка анализа: {result.get('error', 'unknown error')}"
            )
            return

        payload = result.get("result", {})

        confidence = payload.get("confidence")

        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None

        confidence_text = f"{confidence_value:.2f}" if confidence_value is not None else "—"

        await processing_msg.edit_text(
            f"Результат: {payload.get('predicted_class')}\n"
            f"Уверенность: {confidence_text}"
        )

    except Exception as exc:
        logger.exception("Telegram photo inference failed")
        await processing_msg.edit_text(f"Ошибка запуска анализа: {exc}")
    finally:
        local_path.unlink(missing_ok=True)


@router.message(F.document)
async def handle_document(message: Message) -> None:
    document = message.document
    tg_file = await message.bot.get_file(document.file_id)

    local_path = await download_telegram_file(
        bot=message.bot,
        telegram_path=tg_file.file_path,
        filename=document.file_name or f"{document.file_unique_id}.bin",
    )

    processing_msg = await message.answer("Изображение получено, запускаю анализ...")

    try:
        enqueue = await api_client.submit_inference(
            file_path=local_path,
            filename=local_path.name,
            telegram_user_id=message.from_user.id,
            telegram_username=message.from_user.username,
            telegram_display_name=build_display_name(message),
        )

        task_id = enqueue.get("task_id")
        logger.info("Telegram inference task created: task_id=%s", task_id)

        if not task_id:
            await processing_msg.edit_text(
                f"Ошибка запуска анализа: API не вернул task_id.\nОтвет: {enqueue}"
            )
            return

        result = await api_client.wait_for_result(task_id)

        status_value = result.get("status")
        if status_value in {"FAILURE", "failed"}:
            await processing_msg.edit_text(
                f"Ошибка анализа: {result.get('error', 'unknown error')}"
            )
            return

        payload = result.get("result", {})

        confidence = payload.get("confidence")

        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None

        confidence_text = f"{confidence_value:.2f}" if confidence_value is not None else "—"

        await processing_msg.edit_text(
            f"Результат: {payload.get('predicted_class')}\n"
            f"Уверенность: {confidence_text}"
        )

    except Exception as exc:
        logger.exception("Telegram document inference failed")
        await processing_msg.edit_text(f"Ошибка запуска анализа: {exc}")
    finally:
        local_path.unlink(missing_ok=True)