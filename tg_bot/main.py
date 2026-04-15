import asyncio
import logging

from aiogram import Bot, Dispatcher

from tg_bot.config import settings
from tg_bot.handlers.history import router as history_router
from tg_bot.handlers.inference import router as inference_router
from tg_bot.handlers.start import router as start_router


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()

    dp.include_router(start_router)
    dp.include_router(inference_router)
    dp.include_router(history_router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())