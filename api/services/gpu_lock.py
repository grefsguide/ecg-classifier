import redis
from api.core.settings import settings

redis_client = redis.Redis.from_url(settings.celery_broker_url)

GPU_TRAINING_LOCK_KEY = "gpu_training_lock"

def set_gpu_training_lock():
    redis_client.set(GPU_TRAINING_LOCK_KEY, "1")

def clear_gpu_training_lock():
    redis_client.delete(GPU_TRAINING_LOCK_KEY)

def is_gpu_busy_with_training() -> bool:
    return redis_client.exists(GPU_TRAINING_LOCK_KEY) == 1