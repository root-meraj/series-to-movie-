# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE=int8 \
    RENDER_ENCODER=libx264 \
    STS_WEB_HOST=0.0.0.0 \
    STS_WEB_PORT=8765 \
    STS_WEB_JOBS_DIR=/app/data/jobs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates ffmpeg libgl1 libglib2.0-0 libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
# The CPU wheel prevents a CPU deployment from downloading CUDA runtime wheels.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY sts ./sts
RUN python -m pip install --no-deps . \
    && groupadd --gid 10001 sts \
    && useradd --uid 10001 --gid sts --create-home --shell /usr/sbin/nologin sts \
    && mkdir -p /app/data/jobs /models/huggingface /models/torch /models/cache \
    && chown -R sts:sts /app/data /models /home/sts

USER sts
EXPOSE 8765
VOLUME ["/app/data", "/models"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('STS_WEB_PORT', '8765') + '/api/health', timeout=5).read()"]
CMD ["python", "-m", "sts.app.web_api"]
