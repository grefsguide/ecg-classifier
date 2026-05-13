from pathlib import Path

from mlflow.tracking import MlflowClient

from api.core.settings import settings


MLFLOW_RUN_URI_PREFIX = "mlflow://runs/"


def _get_mlflow_client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def build_mlflow_artifact_uri(run_id: str, artifact_path: str) -> str:
    artifact_path = artifact_path.strip("/")
    return f"{MLFLOW_RUN_URI_PREFIX}{run_id}/artifacts/{artifact_path}"


def parse_mlflow_artifact_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith(MLFLOW_RUN_URI_PREFIX):
        raise ValueError(f"Not an MLflow run artifact URI: {uri}")

    rest = uri[len(MLFLOW_RUN_URI_PREFIX):]

    if "/artifacts/" not in rest:
        raise ValueError(
            "Invalid MLflow artifact URI. "
            "Expected format: mlflow://runs/<run_id>/artifacts/<artifact_path>"
        )

    run_id, artifact_path = rest.split("/artifacts/", maxsplit=1)
    if not run_id or not artifact_path:
        raise ValueError(f"Invalid MLflow artifact URI: {uri}")

    return run_id, artifact_path


def log_file_to_mlflow(
    *,
    run_id: str | None,
    local_path: str | Path,
    artifact_dir: str,
) -> str:
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")

    if not run_id:
        return path.resolve().as_uri()

    artifact_dir = artifact_dir.strip("/")
    artifact_path = f"{artifact_dir}/{path.name}"

    client = _get_mlflow_client()
    client.log_artifact(
        run_id=run_id,
        local_path=str(path),
        artifact_path=artifact_dir,
    )

    return build_mlflow_artifact_uri(
        run_id=run_id,
        artifact_path=artifact_path,
    )


def resolve_artifact_uri(
    artifact_uri: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    raw_uri = str(artifact_uri)

    if raw_uri.startswith("file://"):
        return Path(raw_uri.removeprefix("file://"))

    if raw_uri.startswith(MLFLOW_RUN_URI_PREFIX):
        return _download_mlflow_artifact_to_cache(
            artifact_uri=raw_uri,
            cache_dir=cache_dir,
        )

    # Backward compatibility:
    # старые модели могут хранить обычный локальный путь.
    return Path(raw_uri)


def _download_mlflow_artifact_to_cache(
    *,
    artifact_uri: str,
    cache_dir: str | Path | None = None,
) -> Path:
    run_id, artifact_path = parse_mlflow_artifact_uri(artifact_uri)

    cache_root = Path(cache_dir or settings.artifact_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    run_cache_dir = cache_root / run_id
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    expected_local_path = run_cache_dir / artifact_path
    if expected_local_path.exists():
        return expected_local_path

    client = _get_mlflow_client()

    downloaded_path = client.download_artifacts(
        run_id=run_id,
        path=artifact_path,
        dst_path=str(run_cache_dir),
    )

    return Path(downloaded_path)