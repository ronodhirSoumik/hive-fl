"""
Structured logging module for Hive.
Supports text and JSON log output with configurable log levels.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects for machine readability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include custom extra fields if provided
        for key, val in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName"
            ):
                log_entry[key] = val
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with clean timestamp and level tags."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{ts}] [{record.levelname:<7}] [{record.name}]"
        msg = f"{prefix} {record.getMessage()}"
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


def setup_logging(level: str = "INFO", format_type: str = "text") -> None:
    """
    Configures root logger with standard stream handler.
    
    :param level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param format_type: "text" for human-friendly output or "json" for structured logs
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to prevent duplicate messages
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)

    if format_type.lower() == "json":
        stream_handler.setFormatter(JSONFormatter())
    else:
        stream_handler.setFormatter(TextFormatter())

    root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger configured with Hive standard settings."""
    return logging.getLogger(name)