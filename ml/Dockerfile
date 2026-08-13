# ── ML Training container (one-off, not a service) ─────
# Build from project root:
#   docker build -f ml/Dockerfile -t nba-ml-trainer .
#
# Run:
#   docker compose --profile trainer run --rm trainer [--lookback 10]
#
# Optimized with multi-stage build + venv cleanup + .so stripping.

# ── Build stage ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Use Tsinghua mirror for faster downloads in China ─────
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# ── Install CPU-only PyTorch first (separate layer for caching) ──
# CPU wheel is ~192 MB vs ~2.5 GB for CUDA.
# download.pytorch.org S3 endpoint is fast from China (unlike the r2 CDN).
RUN pip install --no-cache-dir --no-compile \
    --default-timeout=600 --retries 5 \
    https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl

# ── Install ML / data packages ───────────────────────────
# onnxscript required for torch.onnx.export (MLP → ONNX for production)
RUN pip install --no-cache-dir --no-compile --default-timeout=300 --retries 5 \
    xgboost \
    numpy \
    pandas \
    scikit-learn \
    optuna \
    asyncpg \
    "sqlalchemy[asyncio]" \
    pydantic-settings \
    kagglehub \
    onnxscript

# ── Clean up venv + strip .so debug symbols ─────────────
# ALL in one RUN so Docker doesn't store intermediate layers
RUN apt-get update && apt-get install -y --no-install-recommends binutils && \
    find /opt/venv -type d \( -name tests -o -name test \) \
        -prune -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name __pycache__ \
        -prune -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name '*.pyc' -delete 2>/dev/null || true && \
    rm -rf /opt/venv/lib/python*/site-packages/pip* 2>/dev/null || true && \
    rm -f /opt/venv/bin/pip* 2>/dev/null || true && \
    find /opt/venv -type f -name '*.so' \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    apt-get remove -y binutils && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ── Runtime stage ────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Prevent Python from writing .pyc files at runtime
ENV PYTHONDONTWRITEBYTECODE=1

# Runtime system deps only (libpq for asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy cleaned + stripped venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ML source
COPY ml/src/ ./ml/src/

# Copy backend ORM models (needed for database queries in feature builder)
COPY backend/app/ ./backend/app/

# Copy scripts
COPY scripts/ ./scripts/

# Create model output directory
RUN mkdir -p /app/ml/saved_models

ENTRYPOINT ["python", "scripts/train_models.py"]
CMD ["--lookback", "10"]
