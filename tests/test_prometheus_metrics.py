import pytest
import requests

from tests.conftest import (
    api_url,
    inference_files,
    prometheus_query,
    wait_for_prometheus_series,
)


@pytest.mark.integration
def test_prometheus_is_available(prometheus_url: str) -> None:
    payload = prometheus_query(prometheus_url, "up")

    result = payload["data"]["result"]
    assert result, payload


@pytest.mark.integration
def test_prometheus_has_api_target(prometheus_url: str) -> None:
    result = wait_for_prometheus_series(
        prometheus_url,
        'up{job="ecg-api"}',
        timeout_seconds=30,
    )

    assert result
    assert result[0]["value"][1] == "1"


@pytest.mark.integration
def test_prometheus_collects_http_metrics(
    api_client: requests.Session,
    prometheus_url: str,
) -> None:
    api_client.get(api_url(api_client, "/health"), timeout=10)

    result = wait_for_prometheus_series(
        prometheus_url,
        "ecg_http_requests_total",
        timeout_seconds=30,
    )

    assert result


@pytest.mark.integration
def test_prometheus_collects_inference_metrics_after_request(
    api_client: requests.Session,
    prometheus_url: str,
    test_image_bytes: bytes,
) -> None:
    response = api_client.post(
        api_url(api_client, "/api/v1/inference/default"),
        files=inference_files(test_image_bytes),
        timeout=240,
    )

    assert response.status_code == 200, response.text

    result = wait_for_prometheus_series(
        prometheus_url,
        'ecg_inference_requests_total{status="success"}',
        timeout_seconds=45,
    )

    assert result


@pytest.mark.integration
def test_prometheus_has_forward_latency_histogram_after_request(
    api_client: requests.Session,
    prometheus_url: str,
    test_image_bytes: bytes,
) -> None:
    response = api_client.post(
        api_url(api_client, "/api/v1/inference/default"),
        files=inference_files(test_image_bytes),
        timeout=240,
    )

    assert response.status_code == 200, response.text

    result = wait_for_prometheus_series(
        prometheus_url,
        "ecg_inference_forward_duration_seconds_bucket",
        timeout_seconds=45,
    )

    assert result