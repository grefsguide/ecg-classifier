# Telegram Bot Documentation

This document describes the Telegram bot integration for the ECG Classifier project.

The Telegram bot allows users to send ECG images through Telegram and receive model predictions without directly using the REST API.

> The project is intended for research and educational purposes. Bot responses are not medical conclusions and must not be used for self-diagnosis or clinical decision-making.

---

## Overview

The Telegram bot is implemented with `aiogram` and communicates with the FastAPI backend.

Main user flow:

```text
User sends ECG photo/file in Telegram
  ↓
Telegram bot downloads the file
  ↓
Bot sends file to FastAPI Telegram inference endpoint
  ↓
FastAPI enqueues Celery inference task
  ↓
Celery worker runs model inference
  ↓
Bot polls task status
  ↓
Bot sends prediction result back to user
```

The bot also supports a history command:

```text
/history
```

It returns the latest inference results for the current Telegram user.

---

## Bot Capabilities

The current Telegram bot supports:

* `/start` command;
* image inference from Telegram photos;
* image inference from Telegram documents/files;
* waiting for asynchronous inference task completion;
* displaying predicted class and confidence;
* `/history` command with recent user results;
* sending previous images back with result captions.

---

## Project Files

Telegram bot code is located in:

```text
tg_bot/
├── main.py
├── config.py
├── handlers/
│   ├── start.py
│   ├── inference.py
│   └── history.py
└── services/
    ├── api_client.py
    └── tg_files.py
```

Related API code:

```text
api/routers/inference_tg.py
api/routers/tg_history.py
api/schemas/tg.py
api/repositories/tg_history.py
```

---

## Environment Variables

The bot uses environment variables from `.env`, Docker Compose, or Kubernetes `ConfigMap` / `Secret`.

Required variables:

| Variable                    | Example           | Description                                       |
| --------------------------- | ----------------- | ------------------------------------------------- |
| `TG_BOT_TOKEN`              | `<token>`         | Telegram bot token from BotFather                 |
| `BOT_API_BASE_URL`          | `http://api:8000` | Internal API URL used by the bot                  |
| `BOT_TEMP_DIR`              | `/tmp/tg-bot`     | Directory for temporary downloaded Telegram files |
| `BOT_POLL_INTERVAL_SECONDS` | `1.5`             | Delay between task status polling attempts        |
| `BOT_POLL_TIMEOUT_SECONDS`  | `180`             | Maximum time to wait for inference result         |

Example `.env` section:

```env
TG_BOT_TOKEN=<token>
BOT_API_BASE_URL=http://api:8000
BOT_TEMP_DIR=/tmp/tg-bot
BOT_POLL_INTERVAL_SECONDS=1.5
BOT_POLL_TIMEOUT_SECONDS=180
```

For Kubernetes, `TG_BOT_TOKEN` should be stored in `Secret`, while non-sensitive bot settings can be stored in `ConfigMap`.

---

## Commands

### `/start`

The `/start` command introduces the bot and explains how to use it.

Current behavior:

```text
Привет! Отправь фото или файл ЭКГ, и я отправлю его на анализ.
Команда /history покажет последние результаты.
```

---

### `/history`

Returns recent inference results for the current Telegram user.

The bot calls:

```http
GET /api/v1/telegram/history/{telegram_user_id}?limit=10
```

For each history item, the bot displays:

* date;
* original file name;
* task status;
* predicted class;
* confidence;
* error message, if any.

If an image URL is available, the bot downloads the image from the API and sends it back with a caption.

---

## Photo and Document Inference

The bot supports two message types:

1. Telegram photos;
2. Telegram documents/files.

### Photo flow

When a user sends a photo:

```text
Telegram photo
  ↓
Bot selects the largest available photo version
  ↓
Bot downloads it locally
  ↓
Bot sends it to API
  ↓
Bot waits for Celery task result
  ↓
Bot returns prediction
```

### Document flow

When a user sends a document:

```text
Telegram document
  ↓
Bot downloads the file locally
  ↓
Bot sends it to API
  ↓
Bot waits for Celery task result
  ↓
Bot returns prediction
```

Both flows use the same API client method:

```python
submit_inference(...)
```

---

## Telegram Inference API

The bot sends files to:

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
| `file`                  | file    |      yes | ECG image file    |
| `telegram_user_id`      | integer |      yes | Telegram user ID  |
| `telegram_username`     | string  |       no | Telegram username |
| `telegram_display_name` | string  |       no | User display name |

Example request:

```bash
curl -X POST "http://localhost:8000/api/v1/inference-tg/default" \
  -F "file=@ecg_img/NORM/IMG_5785.JPG" \
  -F "telegram_user_id=123456789" \
  -F "telegram_username=test_user" \
  -F "telegram_display_name=Test User"
```

Example response:

```json
{
  "task_id": "<TASK_ID>",
  "status": "queued",
  "queue": "infer_gpu",
  "model_key": "default",
  "model_name": "default"
}
```

The Telegram inference endpoint stores:

* uploaded file path;
* Telegram user data;
* task ID;
* queue name;
* initial task status;
* original filename.

---

## Task Polling

After submitting an inference request, the bot polls task status through:

```http
GET /api/v1/tasks/{task_id}
```

Polling settings:

```env
BOT_POLL_INTERVAL_SECONDS=1.5
BOT_POLL_TIMEOUT_SECONDS=180
```

The bot waits until task status becomes one of:

```text
SUCCESS
FAILURE
completed
failed
```

If the task succeeds, the bot reads the result payload and sends:

```text
Результат: <predicted_class>
Уверенность: <confidence>
```

If the task fails, the bot sends an error message.

If the timeout is reached, the bot raises a timeout error and reports the failure to the user.

---

## Telegram History API

### Get user history

```http
GET /api/v1/telegram/history/{telegram_user_id}?limit=10
```

Example:

```bash
curl "http://localhost:8000/api/v1/telegram/history/123456789?limit=10"
```

Example response:

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

### Get history image

```http
GET /api/v1/telegram/history/image/{task_id}
```

Example:

```bash
curl -o history_image.jpg \
  "http://localhost:8000/api/v1/telegram/history/image/<TASK_ID>"
```

The API returns the original image stored for the Telegram inference request.

Supported media type detection:

* `.jpg` / `.jpeg` → `image/jpeg`;
* `.png` → `image/png`;
* `.webp` → `image/webp`;
* `.bmp` → `image/bmp`;
* `.gif` → `image/gif`;
* fallback → `application/octet-stream`.

---

## Local Launch

### Run API stack first

The bot requires a running API, Redis, Celery worker, PostgreSQL, and a registered default model.

With Docker Compose:

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

### Run the bot locally

```bash
poetry run python -m tg_bot.main
```

The bot reads settings from `.env`.

---

## Docker Compose Launch

If the Telegram bot is included in the Docker Compose stack, it should use:

```env
BOT_API_BASE_URL=http://api:8000
```

This is the internal Docker network address of the API service.

Check logs:

```bash
docker compose --env-file .env \
  -f infra/docker-compose.yml \
  -f infra/docker-compose.api.yml \
  logs -f telegram-bot
```

---

## Kubernetes Launch

The Kubernetes deployment uses:

```text
infra/k8s/base/telegram-bot.yaml
```

The bot deployment waits for the API healthcheck before starting.

Required Kubernetes resources:

* `ConfigMap` with bot settings;
* `Secret` with `TG_BOT_TOKEN`;
* API service named `api`;
* shared data PVC mounted at `/shared-data`.

Apply manifests:

```bash
kubectl apply -k infra/k8s/base
```

Check pod:

```bash
kubectl get pods -n ecg-classifier
```

Check logs:

```bash
kubectl logs -n ecg-classifier deploy/telegram-bot -f
```

Restart the bot:

```bash
kubectl rollout restart deployment/telegram-bot -n ecg-classifier
```

---

## Model Selection

The current Telegram inference endpoint uses the default model from the registry.

The bot does not currently expose user-level model selection.

Current behavior:

```text
Telegram image → /api/v1/inference-tg/default → default model
```

To change the model used by the bot, set another model as default through the admin API:

```http
POST /api/v1/admin/models/default
```

Request:

```json
{
  "model_key": "<MODEL_KEY>"
}
```

See `markdown/API.md` for full model registry documentation.

---

## Queue Routing

Telegram inference tasks are routed to Celery queues.

Default behavior:

```text
GPU free → infer_gpu
GPU busy with training → infer_cpu
```

The queue name is stored in Telegram history and returned in the enqueue response.

For model comparison and load testing, use the REST inference API and explicit `model_key` rather than Telegram bot requests.

---

## User-Facing Messages

Current bot messages are in Russian.

### Start message

```text
Привет! Отправь фото или файл ЭКГ, и я отправлю его на анализ.
Команда /history покажет последние результаты.
```

### Processing message

```text
Изображение получено, запускаю анализ...
```

### Success message

```text
Результат: <predicted_class>
Уверенность: <confidence>
```

### Empty history

```text
История пока пуста.
```

### History heading

```text
Последние результаты:
```

---

## Error Handling

The bot handles the following cases:

* API does not return `task_id`;
* inference task fails;
* task polling times out;
* history request fails;
* history image cannot be loaded;
* Telegram file download fails;
* unsupported or invalid image file is sent.

Typical error response to user:

```text
Ошибка запуска анализа: <error>
```

or:

```text
Ошибка анализа: <error>
```

---

## Troubleshooting

### Bot does not start

Check that `TG_BOT_TOKEN` is set:

```bash
echo $TG_BOT_TOKEN
```

In Kubernetes:

```bash
kubectl get secret ecg-secret -n ecg-classifier
kubectl logs -n ecg-classifier deploy/telegram-bot -f
```

---

### Bot cannot connect to API

Check `BOT_API_BASE_URL`.

For Docker Compose and Kubernetes internal networking:

```env
BOT_API_BASE_URL=http://api:8000
```

For local bot outside Docker, use:

```env
BOT_API_BASE_URL=http://localhost:8000
```

Check API health:

```bash
curl http://localhost:8000/health
```

In Kubernetes:

```bash
kubectl exec -n ecg-classifier deploy/telegram-bot -- wget -qO- http://api:8000/health
```

---

### Bot sends request but never receives result

Check Celery workers:

```bash
kubectl logs -n ecg-classifier deploy/worker-gpu -f
kubectl logs -n ecg-classifier deploy/worker-cpu -f
```

Check task status manually:

```bash
curl http://localhost:8000/api/v1/tasks/<TASK_ID>
```

Check Flower:

```bash
kubectl port-forward -n ecg-classifier svc/flower 5555:5555
```

Open:

```text
http://localhost:5555
```

---

### Bot returns model or checkpoint error

Check that a default model is set:

```bash
curl -X GET "http://localhost:8000/api/v1/admin/models/default" \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

If no default model exists, set it through the admin API.

Also check that the checkpoint exists or can be resolved from MLflow/MinIO.

---

### History command fails

Check Telegram history API:

```bash
curl "http://localhost:8000/api/v1/telegram/history/<TELEGRAM_USER_ID>?limit=10"
```

Check that the API can access stored image paths.

If history image URLs are broken, verify the API base URL used for history image generation.

---

## Recommended Validation Flow

1. Start API stack.
2. Run migrations.
3. Register or train at least one model.
4. Set default model.
5. Test REST inference manually.
6. Start Telegram bot.
7. Send an ECG image to the bot.
8. Check prediction result.
9. Run `/history`.
10. Check API, worker, and bot logs if anything fails.

---

## Notes

* The Telegram bot currently uses the default model only.
* Model selection by Telegram command is not implemented yet.
* The bot supports both photos and document uploads.
* Telegram photo compression may affect image quality; for more consistent inference, sending ECG images as documents may be preferable.
* The bot deletes temporary local files after submitting them to the API.
* History images are served by the API from stored upload paths.
