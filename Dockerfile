# Hyperion Dockerfile
# Multi-stage build for production deployment
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime

WORKDIR /app
RUN useradd -m -u 1000 hyperion && \
    mkdir -p /app/data /app/models /app/logs && \
    chown -R hyperion:hyperion /app

COPY --from=builder /root/.local /home/hyperion/.local
COPY . .

ENV PATH=/home/hyperion/.local/bin:$PATH \
    PYTHONPATH=/app \
    HYPERION_CONFIG=/app/config.yaml

USER hyperion
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import hyperion; print('OK')"

ENTRYPOINT ["python", "-m", "hyperion.cli"]