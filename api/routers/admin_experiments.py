from fastapi import APIRouter, Depends, status

from api.core.security import get_current_admin
from api.schemas.experiment import ExperimentCreate, ExperimentTaskResponse
from api.tasks.experiments import run_experiment_task

router = APIRouter(
    prefix="/api/v1/admin/experiments",
    tags=["admin-experiments"],
    dependencies=[Depends(get_current_admin)],
)


@router.post("", response_model=ExperimentTaskResponse, status_code=status.HTTP_202_ACCEPTED)
def create_experiment(payload: ExperimentCreate) -> ExperimentTaskResponse:
    task = run_experiment_task.delay(payload.model_dump())
    return ExperimentTaskResponse(task_id=task.id, status="queued")