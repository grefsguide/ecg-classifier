import uuid
from pathlib import Path

from aiogram import Bot
from aiogram.types import Message

from tg_bot.config import settings


async def download_telegram_file(bot: Bot, telegram_path: str, filename: str) -> Path:
    base_dir = Path(settings.bot_temp_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    local_path = base_dir / f"{uuid.uuid4()}_{filename}"
    await bot.download_file(telegram_path, destination=local_path)
    return local_path


def build_display_name(message: Message) -> str:
    user = message.from_user
    first_name = user.first_name or ""
    last_name = user.last_name or ""
    full_name = f"{first_name} {last_name}".strip()

    if full_name:
        return full_name
    if user.username:
        return f"@{user.username}"

    return f"tg_{user.id}"
