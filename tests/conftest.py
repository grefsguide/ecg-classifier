import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import requests
from PIL import Image, ImageDraw


DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_PROMETHEUS_URL = "http://localhost:9090"


def _strip_trailing_slash(value: str) -> str:
    return value.rstrip("/")


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return _strip_trailing_slash(
        os.getenv("ECG_API_BASE_URL", DEFAULT_API_BASE_URL)
    )


@pytest.fixture(scope="session")
def prometheus_url() -> str:
    return _strip_trailing_slash(
        os.getenv("PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL)
    )


@pytest.fixture(scope="session")
def api_client(api_base_url: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "ecg-classifier-tests/1.0"})
    session.base_url = api_base_url  # type: ignore[attr-defined]
    return session


def api_url(client: requests.Session, path: str) -> str:
    base_url = client.base_url  # type: ignore[attr-defined]
    return f"{base_url}{path}"


def wait_for_http_ok(
    url: str,
    *,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 1.0,
) -> requests.Response:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                return response
        except requests.RequestException as exc:
            last_error = exc

        time.sleep(interval_seconds)

    if last_error is not None:
        raise AssertionError(f"Service is not ready: {url}. Last error: {last_error}")

    raise AssertionError(f"Service is not ready: {url}")


@pytest.fixture(scope="session", autouse=True)
def require_api_ready(api_base_url: str) -> None:
    wait_for_http_ok(f"{api_base_url}/health", timeout_seconds=60)


@pytest.fixture(scope="session")
def admin_credentials() -> tuple[str, str]:
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin")
    return username, password


@pytest.fixture(scope="session")
def admin_headers(
    api_base_url: str,
    admin_credentials: tuple[str, str],
) -> dict[str, str]:
    username, password = admin_credentials

    response = requests.post(
        f"{api_base_url}/api/v1/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )

    if response.status_code != 200:
        pytest.skip(
            "Admin auth is not available. "
            "Set ADMIN_USERNAME and ADMIN_PASSWORD or skip registry tests."
        )

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
def registered_models(
    api_base_url: str,
    admin_headers: dict[str, str],
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{api_base_url}/api/v1/admin/models",
        headers=admin_headers,
        timeout=10,
    )

    if response.status_code != 200:
        pytest.skip(f"Cannot read registered models: {response.text}")

    models = response.json()
    if not models:
        pytest.skip("No registered models found in model registry.")

    return models


@pytest.fixture(scope="session")
def benchmark_model_keys(
    registered_models: list[dict[str, Any]],
) -> list[str]:
    raw_value = os.getenv("TEST_MODEL_KEYS", "").strip()

    if raw_value:
        return [
            item.strip()
            for item in raw_value.split(",")
            if item.strip()
        ]

    return [
        model["model_key"]
        for model in registered_models
        if model.get("is_active", True)
    ]


def _create_synthetic_ecg_png_bytes() -> bytes:
    width = 512
    height = 256

    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    baseline_y = height // 2
    points: list[tuple[int, int]] = []

    for x in range(width):
        cycle = x % 80

        if cycle < 20:
            y = baseline_y
        elif cycle < 28:
            y = baseline_y - int((cycle - 20) * 1.5)
        elif cycle < 36:
            y = baseline_y - int((36 - cycle) * 1.5)
        elif cycle < 40:
            y = baseline_y + 30
        elif cycle < 43:
            y = baseline_y - 70
        elif cycle < 47:
            y = baseline_y + 45
        elif cycle < 62:
            y = baseline_y
        elif cycle < 72:
            y = baseline_y - int((cycle - 62) * 2)
        else:
            y = baseline_y - int((80 - cycle) * 2)

        points.append((x, y))

    draw.line(points, fill="black", width=3)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def test_image_bytes() -> bytes:
    image_path = os.getenv("TEST_IMAGE_PATH")

    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise AssertionError(f"TEST_IMAGE_PATH does not exist: {path}")
        return path.read_bytes()

    return _create_synthetic_ecg_png_bytes()


def inference_files(image_bytes: bytes) -> dict[str, tuple[str, BytesIO, str]]:
    return {
        "file": (
            "synthetic_ecg.png",
            BytesIO(image_bytes),
            "image/png",
        )
    }


def prometheus_query(prometheus_url: str, query: str) -> dict[str, Any]:
    response = requests.get(
        f"{prometheus_url}/api/v1/query",
        params={"query": query},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise AssertionError(f"Prometheus query failed: {payload}")

    return payload


def wait_for_prometheus_series(
    prometheus_url: str,
    query: str,
    *,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] | None = None

    while time.time() < deadline:
        payload = prometheus_query(prometheus_url, query)
        last_payload = payload

        result = payload.get("data", {}).get("result", [])
        if result:
            return result

        time.sleep(interval_seconds)

    raise AssertionError(
        f"Prometheus series not found for query={query}. "
        f"Last payload={last_payload}"
    )