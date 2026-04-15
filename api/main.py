from fastapi import FastAPI

from api.core.settings import settings
from api.routers.admin_experiments import router as admin_experiments_router
from api.routers.admin_models import router as admin_models_router
from api.routers.auth import router as auth_router
from api.routers.inference import router as inference_router
from api.routers.inference_tg import router as inference_tg_router
from api.routers.tasks import router as tasks_router
from api.routers.tg_history import router as telegram_history_router

app = FastAPI(title=settings.app_name)

app.include_router(auth_router)
app.include_router(admin_models_router)
app.include_router(admin_experiments_router)
app.include_router(inference_router)
app.include_router(inference_tg_router)
app.include_router(tasks_router)
app.include_router(telegram_history_router)

@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}