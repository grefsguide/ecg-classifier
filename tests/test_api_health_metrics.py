import pytest
import requests

from tests.conftest import api_url


@pytest.mark.integration
def test_health_endpoint(api_client: requests.Session) -> None:
    response = api_client.get(api_url(api_client, "/health"), timeout=10)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_api_metrics_endpoint_exists(api_client: requests.Session) -> None:
    response = api_client.get(api_url(api_client, "/metrics"), timeout=10)

    assert response.status_code == 200
    assert "ecg_http_requests_total" in response.text
    assert "ecg_http_request_duration_seconds_bucket" in response.text


@pytest.mark.integration
def test_health_request_is_exported_to_metrics(api_client: requests.Session) -> None:
    api_client.get(api_url(api_client, "/health"), timeout=10)

    metrics_response = api_client.get(api_url(api_client, "/metrics"), timeout=10)

    assert metrics_response.status_code == 200
    assert "ecg_http_requests_total" in metrics_response.text