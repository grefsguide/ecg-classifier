# Tests and Load Testing

This document describes the testing strategy for the ECG Classifier project.

The test suite covers:

* API healthcheck;
* API Prometheus metrics;
* default model inference behavior;
* explicit `model_key` inference behavior;
* Prometheus scraping of API and worker metrics;
* load testing by `model_key`;
* model comparison through Prometheus and Grafana.

The tests are designed to validate a running deployment, either with Docker Compose or Kubernetes port-forwarding.

---

## Test Directory Structure

```text
tests/
├── conftest.py
├── test_api_health_metrics.py
├── test_inference_default_model.py
├── test_prometheus_metrics.py
└── load/
    └── test_inference_load.py
```

---

## Test Types

### Integration Tests

Integration tests require a running API, Redis, Celery workers, PostgreSQL, Prometheus, and a registered model.

They check that the service is operational end-to-end.

Examples:

* `GET /health` returns `200`;
* `GET /metrics` is available;
* default inference works without passing `model_key`;
* empty `model_key` falls back to the default model;
* explicit `model_key` inference works;
* Prometheus receives API and worker metrics.

### Load Tests

Load tests generate concurrent inference requests and are used to compare models by:

* throughput;
* p95 total latency;
* p95 forward-only latency;
* queue wait time;
* success/error counts;
* device and queue labels.

Load tests are disabled by default and must be explicitly enabled with `RUN_LOAD_TESTS=true`.

---

## Requirements Before Running Tests

Before running tests, make sure the following services are running:

* API;
* PostgreSQL;
* Redis;
* Celery worker GPU;
* Celery worker CPU;
* Prometheus;
* Grafana, optional for visualization;
* MLflow/MinIO, if training or artifact-based inference is tested.

At least one model must be registered in the model registry, and a default model should be set for default inference tests.

---

## Running Tests with Docker Compose

Start the stack:

```bash
docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.api.yml \
  up --build -d
```

Run migrations:

```bash
docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.api.yml \
  exec api alembic upgrade head
```

Run integration tests:

```bash
poetry run pytest tests -m "integration and not load" -v
```

Run all non-load tests:

```bash
poetry run pytest tests -m "not load" -v
```

---

## Running Tests with Kubernetes

Port-forward the API:

```bash
kubectl port-forward -n ecg-classifier svc/api 8000:8000
```

Port-forward Prometheus:

```bash
kubectl port-forward -n ecg-classifier svc/prometheus 9090:9090
```

Run migrations if needed:

```bash
kubectl exec -n ecg-classifier deploy/api -- alembic upgrade head
```

Run integration tests locally against the port-forwarded services:

```bash
poetry run pytest tests -m "integration and not load" -v
```

If your API or Prometheus runs on a different address, override the URLs:

```bash
ECG_API_BASE_URL=http://localhost:8000 \
PROMETHEUS_URL=http://localhost:9090 \
poetry run pytest tests -m "integration and not load" -v
```

---

## Environment Variables

The test suite can be controlled through environment variables.

### API and Prometheus

| Variable           | Default                 | Description                       |
| ------------------ | ----------------------- | --------------------------------- |
| `ECG_API_BASE_URL` | `http://localhost:8000` | API base URL used by tests        |
| `PROMETHEUS_URL`   | `http://localhost:9090` | Prometheus base URL used by tests |

### Admin Authentication

| Variable         | Default | Description                            |
| ---------------- | ------- | -------------------------------------- |
| `ADMIN_USERNAME` | `admin` | Admin username for protected endpoints |
| `ADMIN_PASSWORD` | `admin` | Admin password for protected endpoints |

If your `.env` uses another password, set it before running tests.

Example:

```bash
ADMIN_USERNAME=admin \
ADMIN_PASSWORD=admin \
poetry run pytest tests -m "integration and not load" -v
```

### Test Image

| Variable          | Default                   | Description                               |
| ----------------- | ------------------------- | ----------------------------------------- |
| `TEST_IMAGE_PATH` | generated synthetic image | Path to ECG image used by inference tests |

If `TEST_IMAGE_PATH` is not provided, the tests generate a simple synthetic ECG-like PNG image.

For more realistic checks, pass a real ECG image:

```bash
TEST_IMAGE_PATH=ecg_img/NORM/IMG_5785.JPG \
poetry run pytest tests -m "integration and not load" -v
```

For Kubernetes-mounted data, use a local file path available to the test process, not a path inside the pod.

---

## Integration Test Files

### `test_api_health_metrics.py`

Checks that the API is alive and exposes Prometheus metrics.

Main checks:

* `GET /health` returns `200`;
* `GET /metrics` returns Prometheus text format;
* API request counters are exported.

Run only this file:

```bash
poetry run pytest tests/test_api_health_metrics.py -v
```

---

### `test_inference_default_model.py`

Checks default model behavior and explicit `model_key` behavior.

Main checks:

* inference works without `model_key`;
* `model_key=""` falls back to the default model;
* `model_key="   "` falls back to the default model;
* inference works for registered model keys.

Run only this file:

```bash
poetry run pytest tests/test_inference_default_model.py -v
```

If no model is registered or no default model is configured, related tests may be skipped or fail depending on the current registry state.

---

### `test_prometheus_metrics.py`

Checks that Prometheus scrapes metrics from API and workers.

Main checks:

* Prometheus is available;
* `ecg-api` target is up;
* API metrics are collected;
* inference worker metrics are collected after an inference request;
* forward latency histogram is available.

Run only this file:

```bash
poetry run pytest tests/test_prometheus_metrics.py -v
```

If these tests fail while inference itself returns `200`, check whether worker metrics are exposed correctly.

Worker metrics endpoints:

```bash
curl http://localhost:9101/metrics
curl http://localhost:9102/metrics
```

Prometheus targets page:

```text
http://localhost:9090/targets
```

Expected targets:

```text
ecg-api
ecg-worker-gpu
ecg-worker-cpu
```

---

## Load Testing

Load tests are located in:

```text
tests/load/test_inference_load.py
```

They are disabled by default. To enable them, set:

```bash
RUN_LOAD_TESTS=true
```

---

## Load Test Parameters

| Variable                       |              Default | Description                                    |
| ------------------------------ | -------------------: | ---------------------------------------------- |
| `RUN_LOAD_TESTS`               |              `false` | Enables load tests                             |
| `TEST_MODEL_KEYS`              |                empty | Comma-separated list of model keys to test     |
| `LOAD_CONCURRENCY`             |                  `4` | Number of concurrent inference requests        |
| `LOAD_REQUESTS_PER_MODEL`      | `20` or `.env` value | Number of requests per explicit model key      |
| `LOAD_REQUEST_TIMEOUT_SECONDS` |                `240` | Timeout for one inference request              |
| `LOAD_INCLUDE_DEFAULT`         |              `false` | Whether to run load test for the default model |
| `LOAD_REQUESTS_DEFAULT_MODEL`  |                 `20` | Number of default-model requests if enabled    |
| `TEST_IMAGE_PATH`              |      synthetic image | Input image used for requests                  |

---

## Running Load Tests

### Minimal load test

```bash
RUN_LOAD_TESTS=true \
TEST_MODEL_KEYS=cnn_v1 \
LOAD_CONCURRENCY=1 \
LOAD_REQUESTS_PER_MODEL=5 \
poetry run pytest tests/load -m load -v -s
```

### Compare several models

```bash
RUN_LOAD_TESTS=true \
TEST_IMAGE_PATH=ecg_img/NORM/IMG_5785.JPG \
TEST_MODEL_KEYS=resnet_20260421_172631_93c40550,vit_20260421_154509_c171c555 \
LOAD_CONCURRENCY=4 \
LOAD_REQUESTS_PER_MODEL=100 \
LOAD_REQUEST_TIMEOUT_SECONDS=300 \
poetry run pytest tests/load -m load -v -s
```

---

## Important Notes About Load Tests

### Use explicit `TEST_MODEL_KEYS`

For model comparison, always pass explicit `TEST_MODEL_KEYS`.

Recommended:

```bash
TEST_MODEL_KEYS=cnn_key,resnet_key,vit_key
```

Avoid relying only on the default model when comparing models, because the default model may change and can hide which model was actually tested.

### Avoid default load test unless needed

The default-model load test is disabled unless:

```bash
LOAD_INCLUDE_DEFAULT=true
```

This prevents extra requests from being added to the default model and skewing model comparison charts.

### Worker concurrency

For local Prometheus metrics to work reliably with Celery workers, the project uses:

```text
--pool=solo --concurrency=1
```

This means one worker processes one task at a time. High `LOAD_CONCURRENCY` values will increase queue wait time.

For model comparison, pay attention to both:

* total inference latency;
* forward-only latency.

Total latency includes queue wait, model loading, preprocessing, forward pass, and postprocessing.

Forward-only latency measures only model execution time.

---

## Prometheus Queries for Test Results

### Successful requests by model

```promql
sum by (model_name, model_key) (
  increase(ecg_inference_requests_total{status="success"}[1h])
)
```

### All requests by model, queue, device, and status

```promql
sum by (model_name, model_key, status, queue, device) (
  increase(ecg_inference_requests_total[1h])
)
```

### Error requests

```promql
sum by (model_name, model_key, queue, device) (
  increase(ecg_inference_requests_total{status="error"}[1h])
)
```

### Throughput by model

```promql
sum by (model_name, model_key) (
  rate(ecg_inference_requests_total{status="success"}[5m])
)
```

### p95 total inference latency

```promql
histogram_quantile(
  0.95,
  sum by (le, model_name, model_key, device) (
    rate(ecg_inference_total_duration_seconds_bucket{status="success"}[5m])
  )
)
```

### p95 forward-only latency

```promql
histogram_quantile(
  0.95,
  sum by (le, model_name, model_key, device) (
    rate(ecg_inference_forward_duration_seconds_bucket{status="success"}[5m])
  )
)
```

### Queue wait p95

```promql
histogram_quantile(
  0.95,
  sum by (le, queue) (
    rate(ecg_inference_queue_wait_seconds_bucket[5m])
  )
)
```

### Distribution by queue and device

```promql
sum by (queue, device) (
  increase(ecg_inference_requests_total[1h])
)
```

---

## Troubleshooting

### API tests fail with connection errors

Check that API is running:

```bash
curl http://localhost:8000/health
```

If using Kubernetes, check port-forwarding:

```bash
kubectl port-forward -n ecg-classifier svc/api 8000:8000
```

---

### Auth tests fail

Check admin credentials:

```bash
ADMIN_USERNAME=admin ADMIN_PASSWORD=admin poetry run pytest tests -v
```

---

### Default inference fails

Check that a default model is configured:

```bash
curl -X GET "http://localhost:8000/api/v1/admin/models/default" \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

If no default model exists, set one:

```bash
curl -X POST "http://localhost:8000/api/v1/admin/models/default" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"model_key": "<MODEL_KEY>"}'
```

---

### Prometheus tests fail but inference returns `200`

This usually means inference works, but worker metrics are not scraped.

Check worker metrics directly:

```bash
curl http://localhost:9101/metrics
curl http://localhost:9102/metrics
```

After an inference request, check:

```bash
curl http://localhost:9101/metrics | grep ecg_inference
curl http://localhost:9102/metrics | grep ecg_inference
```

Also check Prometheus targets:

```text
http://localhost:9090/targets
```

Expected targets:

```text
ecg-api
ecg-worker-gpu
ecg-worker-cpu
```

---

### Only one model appears in Grafana after a load test

Check whether the load test received both model keys.

The test output should print something like:

```text
Parsed TEST_MODEL_KEYS=['resnet_...', 'vit_...']
Running load test for model_key=resnet_...
Running load test for model_key=vit_...
```

Also check all statuses in Prometheus:

```promql
sum by (model_name, model_key, status, queue, device) (
  increase(ecg_inference_requests_total[1h])
)
```

If one model appears only with `status="error"`, inspect worker logs.

---

### All requests go to GPU

This is expected with the default routing strategy:

```text
INFERENCE_QUEUE_STRATEGY=gpu_if_free
```

The CPU queue is used only when GPU is busy with training.

For explicit routing experiments, configure one of:

```text
INFERENCE_QUEUE_STRATEGY=gpu_only
INFERENCE_QUEUE_STRATEGY=cpu_only
INFERENCE_QUEUE_STRATEGY=round_robin
INFERENCE_QUEUE_STRATEGY=gpu_if_free
```

For fair model comparison, prefer `gpu_only` and compare models on the same device.

---

## Recommended Test Flow

### After starting the stack

```bash
poetry run pytest tests/test_api_health_metrics.py -v
```

### After registering a model and setting default

```bash
poetry run pytest tests/test_inference_default_model.py -v
```

### After enabling Prometheus and worker metrics

```bash
poetry run pytest tests/test_prometheus_metrics.py -v
```

### Before model latency comparison

```bash
RUN_LOAD_TESTS=true \
TEST_IMAGE_PATH=ecg_img/NORM/IMG_5785.JPG \
TEST_MODEL_KEYS=<MODEL_KEY_1>,<MODEL_KEY_2> \
LOAD_CONCURRENCY=4 \
LOAD_REQUESTS_PER_MODEL=100 \
poetry run pytest tests/load -m load -v -s
```

Then inspect Grafana dashboards and Prometheus queries.

---

## Notes

* Integration tests validate a running deployment, not isolated unit-level behavior.
* Load tests are intentionally disabled by default.
* For reliable model comparison, pass explicit `TEST_MODEL_KEYS`.
* Use the same `TEST_IMAGE_PATH`, device, queue strategy, and concurrency when comparing models.
* Total latency and forward-only latency measure different things and should not be interpreted as the same metric.
