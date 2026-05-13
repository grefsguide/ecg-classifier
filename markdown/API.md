# API Documentation

This document describes the REST API of the ECG Classifier service.

The API is implemented with FastAPI and is used for:

* authentication;
* model registry management;
* training experiment submission;
* synchronous and asynchronous inference;
* Celery task status inspection;
* Telegram bot integration;
* Telegram inference history;
* service health and metrics.

> For local interactive exploration, use Swagger UI: `http://localhost:8000/docs`.

---

## Base URLs

### Local Docker Compose

```text
http://localhost:8000
```

### Kubernetes Port-forward

```bash
kubectl port-forward -n ecg-classifier svc/api 8000:8000
```

Then use:

```text
http://localhost:8000
```

---

## Authentication

Admin endpoints require a Bearer token.

### Login

```http
POST /api/v1/auth/login
```

Request:

```json
{
  "username": "<YOUR ADMIN NAME>",
  "password": "<YOUR ADMIN PASSWORD>"
}
```

Response:

```json
{
  "access_token": "<JWT_TOKEN>",
  "token_type": "bearer"
}
```

Curl:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin"
  }'
```

Use the returned token in admin requests:

```http
Authorization: Bearer <JWT_TOKEN>
```

---

## Healthcheck

### Check API health

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

Curl:

```bash
curl "http://localhost:8000/health"
```

---

## Model Registry API

Model registry endpoints are protected by admin authentication.

The registry stores model metadata:

* `model_key`;
* `display_name`;
* `model_name`;
* checkpoint path / URI;
* split name;
* MLflow run ID;
* config snapshot;
* metrics;
* tags;
* default model flag.

Supported model names:

```text
cnn
resnet
vit
unet_transformer
```

---

### Register a model manually

```http
POST /api/v1/admin/models
```

Headers:

```http
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json
```

Request:

```json
{
  "model_key": "cnn_v1",
  "display_name": "CNN v1",
  "model_name": "cnn",
  "checkpoint_path": "/app/artifacts/checkpoints/cnn/2026-05-12/10-00-00/cnn.ckpt",
  "checkpoint_uri": "mlflow://runs/<run_id>/artifacts/checkpoints/cnn.ckpt",
  "metrics_uri": "mlflow://runs/<run_id>/artifacts/metrics/cnn_test_metrics.json",
  "storage_backend": "mlflow",
  "split_name": "split_v2",
  "mlflow_run_id": "<run_id>",
  "config_snapshot": {
    "image_size": 224,
    "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
    "batch_size": 16,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001
  },
  "metrics": {
    "test/f1": 0.75,
    "test/recall": 0.76,
    "test/precision": 0.74
  },
  "tags": {
    "source": "manual",
    "kind": "import"
  }
}
```

Curl:

```bash
curl -X POST "http://localhost:8000/api/v1/admin/models" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "cnn_v1",
    "display_name": "CNN v1",
    "model_name": "cnn",
    "checkpoint_path": "/app/artifacts/checkpoints/cnn/2026-05-12/10-00-00/cnn.ckpt",
    "split_name": "split_v2",
    "config_snapshot": {},
    "metrics": {},
    "tags": {"source": "manual"}
  }'
```

> In normal usage, models are registered automatically after training through the experiment API. Manual registration is mainly useful for debugging or importing existing checkpoints.

---

### List registered models

```http
GET /api/v1/admin/models
```

Curl:

```bash
curl -X GET "http://localhost:8000/api/v1/admin/models" \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

Response:

```json
[
  {
    "model_key": "cnn_20260512_100000_abcd1234",
    "display_name": "CNN v1",
    "model_name": "cnn",
    "checkpoint_path": "/app/artifacts/checkpoints/cnn/2026-05-12/10-00-00/cnn.ckpt",
    "checkpoint_uri": "mlflow://runs/<run_id>/artifacts/checkpoints/cnn.ckpt",
    "metrics_uri": "mlflow://runs/<run_id>/artifacts/metrics/cnn_test_metrics.json",
    "storage_backend": "mlflow",
    "split_name": "split_v2",
    "mlflow_run_id": "<run_id>",
    "config_snapshot": {},
    "metrics": {},
    "tags": {},
    "is_active": true,
    "is_default": false,
    "created_at": "2026-05-12T10:00:00",
    "updated_at": "2026-05-12T10:00:00"
  }
]
```

---

### Get default model

```http
GET /api/v1/admin/models/default
```

Curl:

```bash
curl -X GET "http://localhost:8000/api/v1/admin/models/default" \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

If no default model is configured, the API returns `404`.

---

### Set default model

```http
POST /api/v1/admin/models/default
```

Request:

```json
{
  "model_key": "cnn_20260512_100000_abcd1234"
}
```

Curl:

```bash
curl -X POST "http://localhost:8000/api/v1/admin/models/default" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "cnn_20260512_100000_abcd1234"
  }'
```

---

## Training Experiments API

Training is submitted through the API and executed asynchronously by Celery.

```http
POST /api/v1/admin/experiments
```

This endpoint returns a Celery `task_id`. Use the task status endpoint to track progress.

---

### Experiment request schema

Common fields:

```json
{
  "model_name": "cnn",
  "split_name": "split_v2",
  "display_name": "CNN v1",
  "max_epochs": 30,
  "batch_size": 16,
  "img_size": 224,
  "learning_rate": 0.0003,
  "weight_decay": 0.0001,
  "pretrained": true,
  "timm_name": "vit_base_patch16_224",
  "make_default": false,
  "tags": {},
  "extra_overrides": []
}
```

Notes:

* `model_name` selects the model family.
* `split_name` selects the split directory under `artifacts/splits`.
* `display_name` is a human-readable name for the model registry.
* `make_default=true` makes the model the default model after successful training.
* `extra_overrides` is passed to Hydra and should be used for model-specific parameters.
* If a field is not explicitly supported by the API schema, pass it through `extra_overrides`.

Response:

```json
{
  "task_id": "<celery_task_id>",
  "status": "queued"
}
```

---

## Training Examples

### Train CNN

```bash
curl -X POST "http://localhost:8000/api/v1/admin/experiments" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "cnn",
    "split_name": "split_v2",
    "display_name": "CNN v1",
    "max_epochs": 30,
    "batch_size": 16,
    "img_size": 224,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "make_default": false,
    "tags": {
      "source": "api",
      "kind": "train",
      "device": "gpu"
    },
    "extra_overrides": [
      "model.ece_bins=15",
      "model.log_train_prob_metrics=false"
    ]
  }'
```

---

### Train ResNet

```bash
curl -X POST "http://localhost:8000/api/v1/admin/experiments" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet",
    "split_name": "split_v2",
    "display_name": "ResNet18 ImageNet",
    "max_epochs": 30,
    "batch_size": 16,
    "img_size": 224,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "pretrained": true,
    "make_default": false,
    "tags": {
      "source": "api",
      "kind": "train",
      "device": "gpu",
      "backbone_name": "resnet18",
      "pretrained": "imagenet"
    },
    "extra_overrides": [
      "model.backbone_name=resnet18",
      "model.pretrained=true",
      "model.ece_bins=15",
      "model.log_train_prob_metrics=false"
    ]
  }'
```

Possible `backbone_name` values depend on the model factory configuration and available `timm`/torchvision models.

---

### Train ViT

```bash
curl -X POST "http://localhost:8000/api/v1/admin/experiments" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "vit",
    "split_name": "split_v2",
    "display_name": "ViT Base Patch16 224",
    "max_epochs": 30,
    "batch_size": 8,
    "img_size": 224,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "pretrained": true,
    "timm_name": "vit_base_patch16_224",
    "make_default": false,
    "tags": {
      "source": "api",
      "kind": "train",
      "device": "gpu",
      "architecture": "vit"
    },
    "extra_overrides": [
      "model.timm_name=vit_base_patch16_224",
      "model.pretrained=true",
      "model.ece_bins=15",
      "model.log_train_prob_metrics=false"
    ]
  }'
```

---

### Train U-Net + Transformer in latent mode

This mode does not use PTB-XL signal supervision. The model is trained only with classification loss.

```bash
curl -X POST "http://localhost:8000/api/v1/admin/experiments" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "unet_transformer",
    "split_name": "split_v2",
    "display_name": "U-Net + Transformer latent",
    "max_epochs": 30,
    "batch_size": 8,
    "img_size": 224,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "make_default": false,
    "tags": {
      "source": "api",
      "kind": "train",
      "device": "gpu",
      "approach": "B1",
      "supervised_signal": "false"
    },
    "extra_overrides": [
      "model.num_signal_maps=12",
      "model.seq_len=512",
      "model.unet_base_channels=64",
      "model.transformer_d_model=256",
      "model.transformer_nhead=8",
      "model.transformer_num_layers=4",
      "model.transformer_ff_dim=512",
      "model.dropout=0.1",
      "model.softmax_temperature=5.0",
      "model.use_signal_supervision=false",
      "model.ece_bins=15",
      "model.log_train_prob_metrics=false"
    ]
  }'
```

---

### Train supervised U-Net + Transformer

This mode uses ECG images and PTB-XL time-series signals.

Requirements:

* split CSV files must contain `signal_path`;
* `.npy` signal files must exist;
* the number of signal maps should match the number of target leads, usually `12` for PTB-XL;
* for Kubernetes, use `/shared-data/ecg_img` as `data.root_dir`.

```bash
curl -X POST "http://localhost:8000/api/v1/admin/experiments" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "unet_transformer",
    "split_name": "image_series_raw",
    "display_name": "U-Net + Transformer supervised | img384 no_aug temp5",
    "max_epochs": 50,
    "batch_size": 6,
    "img_size": 384,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "make_default": false,
    "tags": {
      "source": "api",
      "kind": "train",
      "device": "gpu",
      "approach": "B2",
      "dataset": "ecg_img",
      "signal_source": "PTB-XL",
      "supervised_signal": "true",
      "architecture": "unet_transformer",
      "experiment": "img384_no_aug_temp5"
    },
    "extra_overrides": [
      "data.root_dir=/shared-data/ecg_img",
      "+data.signal_length=5000",
      "+data.disable_train_augmentations=true",
      "model.ece_bins=15",
      "model.log_train_prob_metrics=false",
      "model.num_signal_maps=12",
      "model.seq_len=1024",
      "model.unet_base_channels=64",
      "model.transformer_d_model=256",
      "model.transformer_nhead=8",
      "model.transformer_num_layers=4",
      "model.transformer_ff_dim=512",
      "model.dropout=0.1",
      "model.softmax_temperature=5.0",
      "model.use_signal_supervision=true",
      "model.signal_loss_weight=0.05"
    ]
  }'
```

For local Docker Compose, the image root may be different. Example:

```json
"data.root_dir=/app/raw_images"
```

For Kubernetes, use:

```json
"data.root_dir=/shared-data/ecg_img"
```

---

## Task Status API

Training and asynchronous inference return Celery task IDs.

### Get task status

```http
GET /api/v1/tasks/{task_id}
```

Curl:

```bash
curl "http://localhost:8000/api/v1/tasks/<TASK_ID>"
```

Response while running:

```json
{
  "task_id": "<TASK_ID>",
  "status": "STARTED"
}
```

Response after success:

```json
{
  "task_id": "<TASK_ID>",
  "status": "SUCCESS",
  "result": {
    "task_id": "<TASK_ID>",
    "status": "completed",
    "model_key": "cnn_20260512_100000_abcd1234",
    "checkpoint_path": "/app/artifacts/checkpoints/cnn/.../cnn.ckpt",
    "checkpoint_uri": "mlflow://runs/<run_id>/artifacts/checkpoints/cnn.ckpt",
    "metrics_uri": "mlflow://runs/<run_id>/artifacts/metrics/cnn_test_metrics.json",
    "storage_backend": "mlflow",
    "mlflow_run_id": "<run_id>",
    "metrics": {},
    "is_default": false
  }
}
```

Response after failure:

```json
{
  "task_id": "<TASK_ID>",
  "status": "FAILURE",
  "error": "<error message>"
}
```

---

## Inference API

Inference can be executed synchronously or asynchronously.

A model can be selected in two ways:

1. leave `model_key` empty — the default model from the registry is used;
2. pass `model_key` — the specified model is used.

Empty values are treated as default model selection:

```text
model_key missing → default model
model_key=""     → default model
model_key="   "  → default model
```

---

### Synchronous inference

```http
POST /api/v1/inference/default
```

Request type:

```text
multipart/form-data
```

Fields:

| Field       | Type   | Required | Description                                         |
| ----------- | ------ | -------: | --------------------------------------------------- |
| `file`      | file   |      yes | ECG image                                           |
| `model_key` | string |       no | specific model key; if empty, default model is used |

Curl with default model:

```bash
curl -X POST "http://localhost:8000/api/v1/inference/default" \
  -F "file=@ecg_img/NORM/IMG_5785.JPG"
```

Curl with explicit model:

```bash
curl -X POST "http://localhost:8000/api/v1/inference/default" \
  -F "file=@ecg_img/NORM/IMG_5785.JPG" \
  -F "model_key=resnet_20260421_172631_93c40550"
```

Response:

```json
{
  "predicted_class": "NORM",
  "confidence": 0.91,
  "probabilities": {
    "CD": 0.02,
    "HYP": 0.01,
    "MI": 0.03,
    "NORM": 0.91,
    "STTC": 0.03
  }
}
```

The synchronous endpoint waits for the Celery task result and may return `504` if the task does not finish within the configured timeout.

---

### Asynchronous inference

```http
POST /api/v1/inference/default/async
```

Request type:

```text
multipart/form-data
```

Fields:

| Field       | Type   | Required | Description                                         |
| ----------- | ------ | -------: | --------------------------------------------------- |
| `file`      | file   |      yes | ECG image                                           |
| `model_key` | string |       no | specific model key; if empty, default model is used |

Curl:

```bash
curl -X POST "http://localhost:8000/api/v1/inference/default/async" \
  -F "file=@ecg_img/NORM/IMG_5785.JPG" \
  -F "model_key=vit_20260421_154509_c171c555"
```

Response:

```json
{
  "task_id": "<TASK_ID>",
  "status": "queued",
  "queue": "infer_gpu"
}
```

Then poll task status:

```bash
curl "http://localhost:8000/api/v1/tasks/<TASK_ID>"
```

---

## Telegram Integration API

These endpoints are intended for the Telegram bot.

---

### Enqueue Telegram inference

```http
POST /api/v1/inference-tg/default
```

Request type:

```text
multipart/form-data
```

Fields:

| Field                   | Type    | Required | Description       |
| ----------------------- | ------- | -------: | ----------------- |
| `file`                  | file    |      yes | ECG image         |
| `telegram_user_id`      | integer |      yes | Telegram user ID  |
| `telegram_username`     | string  |       no | Telegram username |
| `telegram_display_name` | string  |       no | Display name      |

Curl:

```bash
curl -X POST "http://localhost:8000/api/v1/inference-tg/default" \
  -F "file=@ecg_img/NORM/IMG_5785.JPG" \
  -F "telegram_user_id=123456789" \
  -F "telegram_username=test_user" \
  -F "telegram_display_name=Test User"
```

Response:

```json
{
  "task_id": "<TASK_ID>",
  "status": "queued",
  "queue": "infer_gpu",
  "model_key": "default",
  "model_name": "default"
}
```

---

## Telegram History API

### Get Telegram user history

```http
GET /api/v1/telegram/history/{telegram_user_id}?limit=10
```

Curl:

```bash
curl "http://localhost:8000/api/v1/telegram/history/123456789?limit=10"
```

Response:

```json
{
  "items": [
    {
      "task_id": "<TASK_ID>",
      "status": "completed",
      "original_filename": "ecg.jpg",
      "predicted_class": "NORM",
      "confidence": 0.91,
      "probabilities": {
        "CD": 0.02,
        "HYP": 0.01,
        "MI": 0.03,
        "NORM": 0.91,
        "STTC": 0.03
      },
      "error_message": null,
      "created_at": "2026-05-12T10:00:00",
      "image_url": "http://api:8000/api/v1/telegram/history/image/<TASK_ID>"
    }
  ]
}
```

---

### Get image from Telegram history

```http
GET /api/v1/telegram/history/image/{task_id}
```

Curl:

```bash
curl -o history_image.jpg \
  "http://localhost:8000/api/v1/telegram/history/image/<TASK_ID>"
```

---

## Metrics Endpoint

Prometheus scrapes API metrics from:

```http
GET /metrics
```

Curl:

```bash
curl "http://localhost:8000/metrics"
```

Example metrics:

```text
ecg_http_requests_total
ecg_http_request_duration_seconds_bucket
ecg_inference_requests_total
ecg_inference_total_duration_seconds_bucket
ecg_inference_forward_duration_seconds_bucket
ecg_model_load_duration_seconds_bucket
```

Prometheus and Grafana queries are documented in the main `README.md` and test documentation.

---

## Error Handling

Common status codes:

| Status | Meaning                                    |
| -----: | ------------------------------------------ |
|  `200` | Successful synchronous request             |
|  `201` | Model created                              |
|  `202` | Task queued                                |
|  `400` | Bad request, for example empty upload      |
|  `401` | Invalid admin credentials or missing token |
|  `404` | Model/task/default model not found         |
|  `409` | Duplicate `model_key`                      |
|  `504` | Synchronous inference timeout              |
|  `500` | Internal server error                      |

---

## Common Workflows

### Train a model and make it default

1. Login:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "777"}'
```

2. Submit training experiment:

```bash
curl -X POST "http://localhost:8000/api/v1/admin/experiments" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "cnn",
    "split_name": "split_v2",
    "display_name": "CNN v1",
    "max_epochs": 30,
    "batch_size": 16,
    "img_size": 224,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "make_default": true,
    "tags": {"source": "api", "kind": "train"},
    "extra_overrides": [
      "model.ece_bins=15",
      "model.log_train_prob_metrics=false"
    ]
  }'
```

3. Poll task:

```bash
curl "http://localhost:8000/api/v1/tasks/<TASK_ID>"
```

4. Check default model:

```bash
curl -X GET "http://localhost:8000/api/v1/admin/models/default" \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

---

### Run inference with default model

```bash
curl -X POST "http://localhost:8000/api/v1/inference/default" \
  -F "file=@ecg_img/NORM/IMG_5785.JPG"
```

---

### Run inference with a specific model

```bash
curl -X POST "http://localhost:8000/api/v1/inference/default" \
  -F "file=@ecg_img/NORM/IMG_5785.JPG" \
  -F "model_key=resnet_20260421_172631_93c40550"
```

---

## Notes

* Admin endpoints require Bearer authentication.
* Inference endpoints do not require admin authentication by default.
* `model_key` is optional for inference; if omitted, the default model is used.
* Training is asynchronous and executed through Celery.
* For Kubernetes training, use `/shared-data/ecg_img` as image dataset root.
* For supervised `unet_transformer`, split CSV files must contain `signal_path` and signal artifacts must exist.
* Model-specific Hydra settings should be passed through `extra_overrides`.
