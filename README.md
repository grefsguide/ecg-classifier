# ECG Classifier

An DL and MLOps project for ECG image classification and experimental ECG signal extraction from images.

The project covers the full ML lifecycle: data preparation, model training, model registry, asynchronous inference, Telegram bot integration, monitoring, load testing, and Kubernetes deployment.

> This is a research and educational project. Model predictions are not medical conclusions and must not be used for self-diagnosis or clinical decision-making.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Diagnostic Classes](#diagnostic-classes)
* [Modeling Approaches](#modeling-approaches)
* [System Architecture](#system-architecture)
* [Repository Structure](#repository-structure)
* [Technology Stack](#technology-stack)
* [Data and Artifacts](#data-and-artifacts)
* [Environment Variables](#environment-variables)
* [Kubernetes Deployment](#kubernetes-deployment)
* [Initial Setup](#initial-setup)
* [Main Services](#main-services)
* [Model Training](#model-training)
* [Inference](#inference)
* [Monitoring](#monitoring)
* [Telegram Bot](#telegram-bot)
* [Tests and Load Testing](#tests-and-load-testing)
* [Documentation](#documentation)
* [License](#license)

---

## Project Overview

`ecg-classifier` is a service for ECG image analysis using computer vision.

The system receives an ECG image, applies preprocessing, runs the selected model, and returns:

* predicted class;
* class probabilities;
* model confidence;
* execution metadata.

The project supports two main workflows:

1. **Training and model comparison** through API-triggered Celery tasks.
2. **Inference** through REST API and Telegram bot.

---

## Diagnostic Classes

The project uses 5 PTB-XL diagnostic superclasses:

| Class  | Meaning                |
| ------ | ---------------------- |
| `NORM` | Normal ECG             |
| `MI`   | Myocardial Infarction  |
| `STTC` | ST/T Change            |
| `CD`   | Conduction Disturbance |
| `HYP`  | Hypertrophy            |

---

## Modeling Approaches

The project implements two modeling approaches.

### Approach A: Direct ECG Image Classification

```text
ECG image → CNN / ResNet / ViT → class probabilities
```

Supported models:

* `cnn` — baseline CNN;
* `resnet` — ResNet / ResNeXt / WideResNet family;
* `vit` — Vision Transformer through `timm`.

This approach directly classifies ECG images.

### Approach B: ECG Signal Extraction and Sequence Classification

```text
ECG image → U-Net → extracted series → Transformer → class probabilities
```

Model:

* `unet_transformer`.

Approach B supports two modes:

1. **Latent mode** — U-Net extracts a latent time-series representation, and the model is trained only with classification loss.
2. **Supervised signal mode** — if split CSV files contain `signal_path`, the model additionally uses PTB-XL time-series signals and trains with `classification_loss + signal_loss`.

For supervised signal training, `.npy` time-series artifacts from `artifacts/series` are used.

---

## System Architecture

Main production flow:

```text
User / Telegram
  ↓
Telegram Bot / API Client
  ↓
FastAPI
  ↓
Redis Queue
  ↓
Celery Worker
  ↓
Model Registry + Checkpoint Resolver
  ↓
PyTorch Model
  ↓
Prediction + Metrics
```

Components:

* **FastAPI** — REST API for administration, training, inference, model registry, and Telegram request history.
* **Celery** — asynchronous training and inference tasks.
* **Redis** — Celery broker and result backend.
* **PostgreSQL** — users, model registry, default model state, and Telegram history.
* **MLflow** — experiment tracking.
* **MinIO** — S3-compatible artifact storage for MLflow artifacts.
* **Telegram Bot** — user-facing interface for sending ECG images.
* **Prometheus** — technical metrics collection for API and workers.
* **Grafana** — dashboards for latency, throughput, p95, and model comparison.
* **Flower** — Celery task monitoring.
* **Kubernetes** — target deployment environment.

---

## Repository Structure

```text
.
├── alembic/                 # database migrations
├── api/                     # FastAPI app, Celery tasks, services, schemas, registry
├── artifacts/               # local artifacts: splits, checkpoints, metrics, series
├── data_registry/           # service data registry files
├── ecg_classifier/          # ML code: datasets, models, training, Hydra configs
├── infra/                   # Docker and Kubernetes infrastructure
├── markdown/                # extended documentation
├── scripts/                 # helper scripts
├── tests/                   # integration and load tests
├── tg_bot/                  # Telegram bot
├── pyproject.toml
├── alembic.ini
└── README.md
```

---

## Technology Stack

### Backend

* Python 3.11
* FastAPI
* SQLAlchemy
* Alembic
* PostgreSQL
* Redis
* Celery
* Flower

### ML

* PyTorch
* PyTorch Lightning
* TorchMetrics
* torchvision
* timm
* MLflow
* MinIO / S3-compatible storage

### Data

* PTB-XL
* WFDB
* ECG image dataset
* CSV manifests and splits
* `.npy` time-series artifacts

### Infrastructure

* Docker
* Kubernetes
* Kustomize
* Prometheus
* Grafana

---

## Data and Artifacts

For the first Kubernetes deployment, data is loaded from Google Drive by the `data-init` Job.

The deployment expects two `.7z` archives:

```text
PTB-XL.7z
ecg_img.7z
```

After extraction, data is expected to be available in the shared volume:

```text
/shared-data/
├── PTB-XL/
└── ecg_img/
```

The project no longer uses `raw_images` as the main data source in Kubernetes. The main ECG image dataset path is:

```text
/shared-data/ecg_img
```

Training artifacts are stored under:

```text
artifacts/
├── checkpoints/
├── metrics/
├── splits/
└── series/
```

For newly trained models, the source of truth for checkpoints is the PostgreSQL model registry together with MLflow/MinIO artifact URIs. Local checkpoint files remain as fallback and debug artifacts.

---

## Environment Variables

Runtime settings are split between Kubernetes `ConfigMap` and `Secret` resources.

Example local `.env`:

```env
MLFLOW_TRACKING_URI=http://mlflow:8080
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

AWS_ACCESS_KEY_ID=<YOUR MINIO NAME>
AWS_SECRET_ACCESS_KEY=<YOUR MINIO PASSWORD>

JWT_SECRET_KEY=<SECRET_KEY>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=600
ADMIN_USERNAME=<YOUR ADMIN NAME>
ADMIN_PASSWORD=<YOUR ADMMIN PASSWORD>

DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/ecg_classifier

APP_NAME=ecg-classifier-api
API_LOG_LEVEL=INFO

CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

FLOWER_BASIC_AUTH=<FLOWER NAME:FLOWER PASSWORD>
FLOWER_PORT=5555

REGISTRY_DIR=artifacts/registry
TEMP_UPLOAD_DIR=artifacts/uploads

DATASET_NAME=ecg_img
SHARED_DATA_DIR=/shared-data
SHARED_DATASET_DIR=/shared-data/ecg_img
ARTIFACT_CACHE_DIR=/shared-data/model_cache

TG_BOT_TOKEN=<TELEGRAM BOT TOKEN>
BOT_API_BASE_URL=http://api:8000
BOT_TEMP_DIR=/tmp/tg-bot
BOT_POLL_INTERVAL_SECONDS=1.5
BOT_POLL_TIMEOUT_SECONDS=180

PROMETHEUS_ENABLED=true

LOAD_REQUESTS_PER_MODEL=100
LOAD_CONCURRENCY=4
```

---

## Kubernetes Deployment

### 1. Prepare secrets

The repository should contain only a template:

```text
infra/k8s/base/secret.example.yaml
```

Create the real local secret file:

```bash
cp infra/k8s/base/secret.example.yaml infra/k8s/base/secret.yaml
```

Fill in the required values:

* `JWT_SECRET_KEY`
* `ADMIN_PASSWORD`
* `POSTGRES_PASSWORD`
* `MINIO_ROOT_PASSWORD`
* `AWS_SECRET_ACCESS_KEY`
* `TG_BOT_TOKEN`
* `GDRIVE_PTBXL_ARCHIVE_URL`
* `GDRIVE_ECG_IMG_ARCHIVE_URL`

The real `secret.yaml` file is intended for local/deployment use only.

### 2. Build Docker images

For local `minikube`:

```bash
eval $(minikube docker-env)

docker build -t ecg-classifier-app:latest -f infra/docker/Dockerfile.app .
docker build -t ecg-classifier-data-init:latest -f infra/docker/Dockerfile.data-init .
```

For an external cluster:

```bash
docker build -t <registry>/ecg-classifier-app:latest -f infra/docker/Dockerfile.app .
docker build -t <registry>/ecg-classifier-data-init:latest -f infra/docker/Dockerfile.data-init .

docker push <registry>/ecg-classifier-app:latest
docker push <registry>/ecg-classifier-data-init:latest
```

After pushing, update image names in Kubernetes manifests.

### 3. Apply Kubernetes manifests

```bash
kubectl apply -k infra/k8s/base
```

Check the cluster state:

```bash
kubectl get pods -n ecg-classifier
kubectl get jobs -n ecg-classifier
```

### 4. Wait for data initialization

```bash
kubectl logs -n ecg-classifier job/data-init -f
```

Check extracted data:

```bash
kubectl exec -n ecg-classifier deploy/api -- ls -lah /shared-data
kubectl exec -n ecg-classifier deploy/api -- ls -lah /shared-data/ecg_img
kubectl exec -n ecg-classifier deploy/api -- ls -lah /shared-data/PTB-XL
```

### 5. Run database migrations

```bash
kubectl exec -n ecg-classifier deploy/api -- alembic upgrade head
```

---

## Initial Setup

After the cluster starts, check resources:

```bash
kubectl get pods -n ecg-classifier
kubectl get svc -n ecg-classifier
```

For local access, use port-forwarding.

API:

```bash
kubectl port-forward -n ecg-classifier svc/api 8000:8000
```

MLflow:

```bash
kubectl port-forward -n ecg-classifier svc/mlflow 8080:8080
```

MinIO:

```bash
kubectl port-forward -n ecg-classifier svc/minio 9000:9000 9001:9001
```

Flower:

```bash
kubectl port-forward -n ecg-classifier svc/flower 5555:5555
```

Prometheus:

```bash
kubectl port-forward -n ecg-classifier svc/prometheus 9090:9090
```

Grafana:

```bash
kubectl port-forward -n ecg-classifier svc/grafana 3000:3000
```

---

## Main Services

| Service    | Purpose                        | Port-forward   |
| ---------- | ------------------------------ | -------------- |
| API        | REST API / Swagger / inference | `8000`         |
| MLflow     | Experiment tracking            | `8080`         |
| MinIO      | S3-compatible artifact storage | `9000`, `9001` |
| Flower     | Celery monitoring              | `5555`         |
| Prometheus | Metrics storage                | `9090`         |
| Grafana    | Dashboards                     | `3000`         |

Swagger UI after port-forwarding:

```text
http://localhost:8000/docs
```

Healthcheck:

```bash
curl http://localhost:8000/health
```

---

## Model Training

Training is triggered through the API and executed asynchronously by Celery.

Supported models:

* `cnn`
* `resnet`
* `vit`
* `unet_transformer`

Detailed API examples are available in [API markdown](markdown/API.md)

---

## Inference

Inference can be performed through the REST API or the Telegram bot.

By default, the model marked as default in the model registry is used.

A specific model can also be selected by passing `model_key`. This allows multiple models to be compared without creating separate inference endpoints.

```text
POST /api/v1/inference/default
```

See details in [API markdown](markdown/API.md)

---

## Monitoring

The project exposes technical metrics through Prometheus.

Main metric groups:

* API HTTP latency;
* request throughput;
* inference latency;
* forward-only latency;
* queue wait time;
* model load latency;
* cache hit / cache miss;
* p95 by model;
* comparison by `model_name` and `model_key`.

Example throughput query:

```promql
sum by (model_name, model_key) (
  rate(ecg_inference_requests_total{status="success"}[5m])
)
```

Example p95 total latency query:

```promql
histogram_quantile(
  0.95,
  sum by (le, model_name, model_key) (
    rate(ecg_inference_total_duration_seconds_bucket{status="success"}[5m])
  )
)
```

Grafana uses Prometheus as the default datasource.

---

## Telegram Bot

The Telegram bot receives ECG images, sends them to the API, waits for the inference task to complete, and returns the prediction to the user.

Detailed Telegram bot documentation is available in [TG-bot markdown](markdown/TG.md)


---

## Tests and Load Testing

Integration tests and load testing scenarios are located in:

```text
tests/
```

The full test documentation is available in [Test markdown](markdown/TESTS.md)


---

## Documentation

This README describes the project and deployment flow.

Detailed documentation is stored separately:

```text
markdown/API.md    # API, auth, experiments, inference, model registry
markdown/TG.md     # Telegram bot
markdown/TESTS.md  # integration tests and load testing
```

---
## Project tested localy on
- CPU: Intel i9-12900HK
- GPU: RTX 3080 Ti Laptop
- RAM: 64 GB


