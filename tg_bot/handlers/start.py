from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router()


@router.message(CommandStart())
async def start_command(message: Message) -> None:
    await message.answer(
        "Привет! Отправь фото или файл ЭКГ, и я отправлю его на анализ.\n"
        "Команда /history покажет последние результаты."
    )