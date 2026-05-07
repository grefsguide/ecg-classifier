from time import perf_counter
from fastapi import FastAPI, Request
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app, start_http_server
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

HTTP_REQUESTS_TOTAL = Counter(
    "ecg_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "ecg_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "ecg_inference_requests_total",
    "Total inference requests handled by workers.",
    ["model_name", "model_key", "queue", "device", "source", "status"],
)

INFERENCE_QUEUE_WAIT_SECONDS = Histogram(
    "ecg_inference_queue_wait_seconds",
    "Time spent waiting in queue before worker started processing.",
    ["queue", "source"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

INFERENCE_TOTAL_DURATION_SECONDS = Histogram(
    "ecg_inference_total_duration_seconds",
    "Total inference task duration on worker.",
    ["model_name", "model_key", "queue", "device", "source", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 60.0, 120.0),
)

INFERENCE_FORWARD_DURATION_SECONDS = Histogram(
    "ecg_inference_forward_duration_seconds",
    "Pure model forward latency.",
    ["model_name", "model_key", "device", "source", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODEL_LOAD_DURATION_SECONDS = Histogram(
    "ecg_model_load_duration_seconds",
    "Checkpoint/model load duration.",
    ["model_name", "model_key", "device", "cache_hit"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

MODEL_CACHE_ENTRIES = Gauge(
    "ecg_model_cache_entries",
    "Number of cached models inside current worker process.",
)

MODEL_CACHE_HITS_TOTAL = Counter(
    "ecg_model_cache_hits_total",
    "Model cache hits.",
    ["model_name", "model_key", "device"],
)

MODEL_CACHE_MISSES_TOTAL = Counter(
    "ecg_model_cache_misses_total",
    "Model cache misses.",
    ["model_name", "model_key", "device"],
)


class PrometheusHttpMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        started = perf_counter()
        response = await call_next(request)
        elapsed = perf_counter() - started

        path = request.url.path
        method = request.method
        status_code = str(response.status_code)

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            path=path,
            status_code=status_code,
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            path=path,
            status_code=status_code,
        ).observe(elapsed)

        return response


def mount_metrics_endpoint(app: FastAPI) -> None:
    app.mount("/metrics", make_asgi_app())


def start_worker_metrics_server(port: int) -> None:
    start_http_server(port)


def observe_queue_wait(queue: str, source: str, seconds: float) -> None:
    INFERENCE_QUEUE_WAIT_SECONDS.labels(queue=queue, source=source).observe(seconds)


def observe_inference_total(
    *,
    model_name: str,
    model_key: str,
    queue: str,
    device: str,
    source: str,
    status: str,
    seconds: float,
) -> None:
    INFERENCE_TOTAL_DURATION_SECONDS.labels(
        model_name=model_name,
        model_key=model_key,
        queue=queue,
        device=device,
        source=source,
        status=status,
    ).observe(seconds)


def observe_inference_forward(
    *,
    model_name: str,
    model_key: str,
    device: str,
    source: str,
    status: str,
    seconds: float,
) -> None:
    INFERENCE_FORWARD_DURATION_SECONDS.labels(
        model_name=model_name,
        model_key=model_key,
        device=device,
        source=source,
        status=status,
    ).observe(seconds)


def observe_model_load(
    *,
    model_name: str,
    model_key: str,
    device: str,
    cache_hit: bool,
    seconds: float,
) -> None:
    MODEL_LOAD_DURATION_SECONDS.labels(
        model_name=model_name,
        model_key=model_key,
        device=device,
        cache_hit=str(cache_hit).lower(),
    ).observe(seconds)


def inc_inference_requests_total(
    *,
    model_name: str,
    model_key: str,
    queue: str,
    device: str,
    source: str,
    status: str,
) -> None:
    INFERENCE_REQUESTS_TOTAL.labels(
        model_name=model_name,
        model_key=model_key,
        queue=queue,
        device=device,
        source=source,
        status=status,
    ).inc()