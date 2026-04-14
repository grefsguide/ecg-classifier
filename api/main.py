from fastapi import FastAPI

from api.core.settings import settings
from api.routers.admin_experiments import router as admin_experiments_router
from api.routers.admin_models import router as admin_models_router
from api.routers.auth import router as auth_router
from api.routers.inference import router as inference_router
from api.routers.inference_test import router as inference_test

app = FastAPI(title=settings.app_name)

app.include_router(auth_router)
app.include_router(admin_models_router)
app.include_router(admin_experiments_router)
app.include_router(inference_router)
app.include_router(inference_test)

@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}