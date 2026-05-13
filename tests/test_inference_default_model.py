import pytest
import requests

from tests.conftest import api_url, inference_files


def _assert_valid_inference_response(payload: dict) -> None:
    assert "predicted_class" in payload
    assert "confidence" in payload
    assert "probabilities" in payload

    assert isinstance(payload["predicted_class"], str)
    assert isinstance(payload["confidence"], float | int)
    assert isinstance(payload["probabilities"], dict)

    assert 0.0 <= float(payload["confidence"]) <= 1.0
    assert len(payload["probabilities"]) > 0


@pytest.mark.integration
def test_inference_uses_default_model_when_model_key_is_missing(
    api_client: requests.Session,
    test_image_bytes: bytes,
) -> None:
    response = api_client.post(
        api_url(api_client, "/api/v1/inference/default"),
        files=inference_files(test_image_bytes),
        timeout=240,
    )

    assert response.status_code == 200, response.text
    _assert_valid_inference_response(response.json())


@pytest.mark.integration
def test_inference_uses_default_model_when_model_key_is_empty(
    api_client: requests.Session,
    test_image_bytes: bytes,
) -> None:
    response = api_client.post(
        api_url(api_client, "/api/v1/inference/default"),
        files=inference_files(test_image_bytes),
        data={"model_key": ""},
        timeout=240,
    )

    assert response.status_code == 200, response.text
    _assert_valid_inference_response(response.json())


@pytest.mark.integration
def test_inference_uses_default_model_when_model_key_is_blank(
    api_client: requests.Session,
    test_image_bytes: bytes,
) -> None:
    response = api_client.post(
        api_url(api_client, "/api/v1/inference/default"),
        files=inference_files(test_image_bytes),
        data={"model_key": "   "},
        timeout=240,
    )

    assert response.status_code == 200, response.text
    _assert_valid_inference_response(response.json())


@pytest.mark.integration
def test_inference_by_registered_model_key(
    api_client: requests.Session,
    test_image_bytes: bytes,
    benchmark_model_keys: list[str],
) -> None:
    for model_key in benchmark_model_keys:
        response = api_client.post(
            api_url(api_client, "/api/v1/inference/default"),
            files=inference_files(test_image_bytes),
            data={"model_key": model_key},
            timeout=240,
        )

        assert response.status_code == 200, (
            f"model_key={model_key}; response={response.text}"
        )
        _assert_valid_inference_response(response.json())