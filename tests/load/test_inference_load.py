import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from statistics import mean
from time import perf_counter

import pytest
import requests


def _parse_model_keys() -> list[str]:
    raw_value = os.getenv("TEST_MODEL_KEYS", "").strip()

    if not raw_value:
        return []

    model_keys = [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]

    return model_keys


LOAD_MODEL_KEYS = _parse_model_keys()


def _build_files(image_bytes: bytes) -> dict[str, tuple[str, BytesIO, str]]:
    return {
        "file": (
            "load_test_ecg.png",
            BytesIO(image_bytes),
            "image/png",
        )
    }


def _run_single_inference(
    *,
    api_base_url: str,
    image_bytes: bytes,
    model_key: str | None,
    timeout_seconds: int,
) -> tuple[int, float, str]:
    started = perf_counter()

    data: dict[str, str] = {}
    if model_key:
        data["model_key"] = model_key

    response = requests.post(
        f"{api_base_url}/api/v1/inference/default",
        files=_build_files(image_bytes),
        data=data,
        timeout=timeout_seconds,
    )

    elapsed = perf_counter() - started
    return response.status_code, elapsed, response.text


def _run_load_for_model(
    *,
    api_base_url: str,
    image_bytes: bytes,
    model_key: str | None,
    concurrency: int,
    requests_count: int,
    timeout_seconds: int,
) -> None:
    latencies: list[float] = []
    errors: list[str] = []

    print()
    print("=" * 80)
    print(f"Running load test for model_key={model_key or '<default>'}")
    print(f"requests_count={requests_count}")
    print(f"concurrency={concurrency}")
    print(f"timeout_seconds={timeout_seconds}")
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _run_single_inference,
                api_base_url=api_base_url,
                image_bytes=image_bytes,
                model_key=model_key,
                timeout_seconds=timeout_seconds,
            )
            for _ in range(requests_count)
        ]

        for future in as_completed(futures):
            status_code, elapsed, response_text = future.result()
            latencies.append(elapsed)

            if status_code != 200:
                errors.append(
                    f"status_code={status_code}; response={response_text[:500]}"
                )

    assert not errors, (
        f"model_key={model_key}; "
        f"errors_count={len(errors)}; "
        f"first_error={errors[0] if errors else None}"
    )

    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
    p95 = sorted_latencies[p95_index]

    print()
    print(
        f"model_key={model_key or '<default>'}; "
        f"requests={requests_count}; "
        f"concurrency={concurrency}; "
        f"avg_latency={mean(latencies):.4f}s; "
        f"p95_latency={p95:.4f}s; "
        f"min_latency={min(latencies):.4f}s; "
        f"max_latency={max(latencies):.4f}s"
    )


@pytest.mark.integration
@pytest.mark.load
def test_load_env_model_keys_are_parsed() -> None:
    if os.getenv("RUN_LOAD_TESTS", "false").lower() != "true":
        pytest.skip("Set RUN_LOAD_TESTS=true to run load tests.")

    assert LOAD_MODEL_KEYS, (
        "TEST_MODEL_KEYS is empty. "
        "For load tests pass explicit model keys, for example: "
        "TEST_MODEL_KEYS=resnet_...,vit_..."
    )

    print()
    print(f"Parsed TEST_MODEL_KEYS={LOAD_MODEL_KEYS}")


@pytest.mark.integration
@pytest.mark.load
@pytest.mark.parametrize("model_key", LOAD_MODEL_KEYS)
def test_inference_load_by_model_key(
    api_base_url: str,
    test_image_bytes: bytes,
    model_key: str,
) -> None:
    if os.getenv("RUN_LOAD_TESTS", "false").lower() != "true":
        pytest.skip("Set RUN_LOAD_TESTS=true to run load tests.")

    concurrency = int(os.getenv("LOAD_CONCURRENCY", "4"))
    requests_per_model = int(os.getenv("LOAD_REQUESTS_PER_MODEL", "20"))
    timeout_seconds = int(os.getenv("LOAD_REQUEST_TIMEOUT_SECONDS", "240"))

    assert concurrency > 0
    assert requests_per_model > 0

    _run_load_for_model(
        api_base_url=api_base_url,
        image_bytes=test_image_bytes,
        model_key=model_key,
        concurrency=concurrency,
        requests_count=requests_per_model,
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.integration
@pytest.mark.load
def test_inference_load_default_model(
    api_base_url: str,
    test_image_bytes: bytes,
) -> None:
    if os.getenv("RUN_LOAD_TESTS", "false").lower() != "true":
        pytest.skip("Set RUN_LOAD_TESTS=true to run load tests.")

    if os.getenv("LOAD_INCLUDE_DEFAULT", "false").lower() != "true":
        pytest.skip("Set LOAD_INCLUDE_DEFAULT=true to load test default model.")

    concurrency = int(os.getenv("LOAD_CONCURRENCY", "4"))
    requests_count = int(os.getenv("LOAD_REQUESTS_DEFAULT_MODEL", "20"))
    timeout_seconds = int(os.getenv("LOAD_REQUEST_TIMEOUT_SECONDS", "240"))

    _run_load_for_model(
        api_base_url=api_base_url,
        image_bytes=test_image_bytes,
        model_key=None,
        concurrency=concurrency,
        requests_count=requests_count,
        timeout_seconds=timeout_seconds,
    )