from celery import Celery

from app.config import settings

celery_app = Celery(
    "engram",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    beat_schedule={
        "compress-daily": {
            "task": "app.workers.tasks.run_compression_pipeline",
            "schedule": 86400.0,  # Level 2/3/4 aging job, once a day
        },
    },
)
