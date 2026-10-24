FROM python:3.10-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 1. Install heavy dependencies first (Layer-cached)
# Using PyTorch CPU index reduces download from ~2.8GB to ~180MB (~15x speedup)
COPY pyproject.toml README.md LICENSE ./
RUN uv pip install --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "fastapi>=0.100.0" \
    "uvicorn[standard]>=0.23.0" \
    "httpx>=0.24.0" \
    "pydantic>=2.0.0" \
    "typer>=0.9.0" \
    "torch>=2.0.0" \
    "numpy>=1.24.0" \
    "safetensors>=0.4.0" \
    "cryptography>=41.0.0" \
    "psutil>=5.9.0" \
    "scipy>=1.10.0" \
    "python-multipart>=0.0.6"

# 2. Copy source code and install hive-fl without re-downloading dependencies
# Any changes to src/ build in under 2 seconds!
COPY src/ ./src/
RUN uv pip install --system --no-deps -e .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 9000 8765 8766

ENTRYPOINT ["hive"]
