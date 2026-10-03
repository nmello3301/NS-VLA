# NS-VLA JAX inference wrapper
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NSVLA_HOST=0.0.0.0 \
    NSVLA_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /app/docker/requirements.txt
RUN pip install -r /app/docker/requirements.txt

COPY nsvla /app/nsvla
COPY setup.py /app/setup.py
COPY docker /app/docker

ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "docker.server:app", "--host", "0.0.0.0", "--port", "8000"]
