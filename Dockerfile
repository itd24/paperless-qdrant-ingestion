# One-shot Paperless -> Qdrant ingestion worker.
#
# Deliberately contains no cron and no scheduler: the container runs a single
# ingestion and exits. Scheduling lives on the host, e.g. a cron entry doing
# `docker compose run --rm paperless-ingestion` hourly.
FROM python:3.12-slim

WORKDIR /app

# The tokenizer needs to download its model files on first run unless HF_HOME
# is pointed at the mounted cache.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/hf-cache \
    PIP_NO_CACHE_DIR=1

COPY requirements-ingestion.txt .
RUN pip install --no-cache-dir -r requirements-ingestion.txt

COPY ingest_paperless.py .

# Read-only tag->domain mapping, writable state/checkpoint directory.
# Both come from bind mounts in document-ingestion.yml.
VOLUME ["/app/data"]

ENTRYPOINT ["python", "ingest_paperless.py"]
