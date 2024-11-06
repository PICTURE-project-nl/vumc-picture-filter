import os

# Redis.
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Celery.
CELERY_CONFIG = {
  "broker_url": REDIS_URL,
  "result_backend": REDIS_URL,
  "include": []
}
