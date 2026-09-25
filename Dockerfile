# Agent server only - llama-server/llama-swap and the models run on another
# machine (LLAMA_BACKEND=llama-swap, LLAMA_SERVER_URL in .env). No GPU,
# no CUDA, no dev dependencies in here. See docker-compose.yml for how
# it's run and what gets mounted.
FROM python:3.12-slim

# tzdata: the agent injects datetime.now() into every prompt, and a
# container is UTC unless TZ is set (docker-compose.yml sets it) AND the
# zone files exist.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    DATA_ROOT=/data \
    LLAMA_BACKEND=llama-swap

WORKDIR /app

# CPU-only torch first: memory.py runs sentence-transformers on CPU, and the
# default torch wheel would drag in several GB of CUDA libraries for nothing.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the embedding model into the image so startup never downloads it.
# Keep in sync with config.EMBEDDING_MODEL.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"
ENV HF_HUB_OFFLINE=1

COPY . .

RUN useradd --create-home --uid 1000 agent \
 && mkdir -p /data \
 && chown -R agent:agent /data
USER agent

EXPOSE 8100

# /model/status answers 200 whether or not the remote model machine is up -
# this only checks the agent itself, so a sleeping GPU box doesn't make
# Docker restart the agent in a loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/model/status', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
