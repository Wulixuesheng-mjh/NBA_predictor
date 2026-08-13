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

# ── Layer 1: install heavy ML packages first (cached separately) ──
# These rarely change — keeping them in a dedicated layer avoids re-downloads
# when only app code or lightweight deps change.
# onnxruntime provides MLP inference without needing PyTorch (~10 MB vs ~800 MB).
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-compile --default-timeout=300 --retries 5 \
    xgboost>=2.0 shap>=0.44.0 numpy>=1.26.0 scikit-learn>=1.3.0 onnxruntime>=1.18.0

# ── Layer 2: copy app source + install the app + remaining deps ──
# `app/` must be present for `pip install "."` to find the package.
# Alembic also needs the migrations directory at runtime.
COPY app/ app/
COPY alembic.ini .
COPY alembic/ alembic/

# Using "." (NOT ".[dev]") to exclude pytest, aiosqlite from production image.
RUN pip install --no-cache-dir --no-compile --default-timeout=120 "." && \
    pip install --no-cache-dir --no-compile alembic

# ── Layer 3: copy remaining files (config, scripts, etc.) ──
COPY . .

# ── Clean up venv + strip .so debug symbols ─────────────
# ALL in one RUN so Docker doesn't store intermediate layers:
# 1. Install binutils for strip, then remove it
# 2. Delete test dirs (~100 MB), __pycache__, .pyc, pip
# 3. Strip debug symbols from .so files (~576→460 MB measured)
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

# Prevent Python from writing .pyc files at runtime (Docker containers are ephemeral)
ENV PYTHONDONTWRITEBYTECODE=1

# Runtime system deps only (libpq for asyncpg, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy venv (cleaned + stripped) from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code (alembic.ini + migrations needed at runtime for `alembic upgrade head`)
COPY --chown=appuser:appuser . .

# Create directory for ML source (mounted at runtime in dev, or copied in prod build)
RUN mkdir -p /app/ml/src /app/ml/saved_models && touch /app/ml/__init__.py

# Drop privileges
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run migrations then start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info"]
