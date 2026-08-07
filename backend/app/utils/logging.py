"""Structured JSON logging for the NBA Predictor backend.

Outputs one JSON object per line to stdout, compatible with
log aggregators (ELK, Loki, Datadog, etc.) and `docker logs`.

Usage:
    from app.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Prediction generated", extra={"home_team": "LAL", "duration_ms": 42})
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ── Context variables (propagate through async tasks) ──
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
request_path_var: ContextVar[str] = ContextVar("request_path", default="")


def generate_request_id() -> str:
    """Generate a short unique request ID."""
    return uuid.uuid4().hex[:12]


# ── JSON Formatter ──────────────────────────────────────


class StructuredFormatter(logging.Formatter):
    """Format log records as JSON with automatic timestamp and level."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include error info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields passed via `extra=` kwarg
        for key in ("duration_ms", "status_code", "method", "path",
                     "team_id", "game_id", "model_name",
                     "records_collected", "task_name", "source",
                     "prediction_id", "error_type"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # Include any other custom extras except standard LogRecord attrs
        standard_attrs = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "msg", "name", "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                if key not in log_entry:
                    log_entry[key] = value

        # Inject async context
        req_id = request_id_var.get()
        if req_id:
            log_entry["request_id"] = req_id
        req_path = request_path_var.get()
        if req_path:
            log_entry["path"] = req_path

        return json.dumps(log_entry, ensure_ascii=False, default=str)


# ── Setup ───────────────────────────────────────────────


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured JSON output.

    Called once at application startup (in main.py lifespan).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger()
    # Clear any existing handlers to avoid duplicate output
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("nba_api").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance configured for structured output.

    Convenience wrapper around logging.getLogger.
    """
    return logging.getLogger(name)


def log_duration(logger: logging.Logger, message: str,
                 start_time: float, **extra: Any) -> None:
    """Log a message with an automatic duration_ms field.

    Usage:
        t0 = time.monotonic()
        # ... do work ...
        log_duration(logger, "Training complete", t0,
                     model="xgboost", samples=1200)
    """
    duration_ms = round((time.monotonic() - start_time) * 1000, 1)
    logger.info(message, extra={"duration_ms": duration_ms, **extra})
