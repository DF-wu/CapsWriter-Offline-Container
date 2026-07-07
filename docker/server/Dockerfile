FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04@sha256:85fb7ac694079fff1061a0140fd5b5a641997880e12112d92589c3bbb1e8b7ca

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    CAPSWRITER_MODEL_TYPE=qwen_asr \
    CAPSWRITER_ENABLE_TRAY=false \
    CAPSWRITER_SERVER_ADDR=0.0.0.0 \
    CAPSWRITER_SERVER_PORT=6016 \
    CAPSWRITER_LOG_DIR=/app/logs \
    CAPSWRITER_INTERACTIVE_ERRORS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip python3-tk tini libgomp1 libvulkan1 ca-certificates ffmpeg \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install \
        packaging==26.2 \
        pip==26.1.2 \
        setuptools==83.0.0 \
        wheel==0.47.0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server-docker.txt requirements-server-docker.lock /app/

RUN python -m pip install --no-build-isolation -r /app/requirements-server-docker.lock

COPY . /app

RUN mkdir -p /app/logs /app/models \
    && useradd --create-home --shell /bin/bash appuser \
    && chmod +x /app/docker/server/entrypoint.sh \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 6016 6017

HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=5 \
  CMD ["python", "docker/server/healthcheck.py"]

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/server/entrypoint.sh"]
