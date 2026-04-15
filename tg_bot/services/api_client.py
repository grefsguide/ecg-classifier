import asyncio
import logging
from pathlib import Path

import httpx

from tg_bot.config import settings

logger = logging.getLogger(__name__)


class ApiClient:
    async def submit_inference(
        self,
        *,
        file_path: Path,
        filename: str,
        telegram_user_id: int,
        telegram_username: str | None,
        telegram_display_name: str | None,
    ) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            with file_path.open("rb") as f:
                response = await client.post(
                    f"{settings.api_base_url}/api/v1/inference-tg/default",
                    files={"file": (filename, f, "application/octet-stream")},
                    data={
                        "telegram_user_id": str(telegram_user_id),
                        "telegram_username": telegram_username or "",
                        "telegram_display_name": telegram_display_name or "",
                    },
                )

            logger.info(
                "submit_inference response: status=%s body=%s",
                response.status_code,
                response.text,
            )

            response.raise_for_status()
            return response.json()

    async def get_task_status(self, task_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{settings.api_base_url}/api/v1/tasks/{task_id}")

            logger.info(
                "get_task_status response: task_id=%s status=%s body=%s",
                task_id,
                response.status_code,
                response.text,
            )

            response.raise_for_status()
            return response.json()

    async def wait_for_result(self, task_id: str) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.poll_timeout_seconds

        while True:
            payload = await self.get_task_status(task_id)
            status_value = payload.get("status")

            if status_value in {"SUCCESS", "FAILURE", "completed", "failed"}:
                return payload

            if loop.time() > deadline:
                raise TimeoutError(f"Task {task_id} timed out")

            await asyncio.sleep(settings.poll_interval_seconds)

    async def get_history(self, telegram_user_id: int, limit: int = 10) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.api_base_url}/api/v1/telegram/history/{telegram_user_id}",
                params={"limit": limit},
            )
            response.raise_for_status()
            return response.json()