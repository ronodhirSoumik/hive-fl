"""
Unit tests for structured logging (JSON and text formats).
"""

import json
import logging
from hive.core.logging import JSONFormatter, TextFormatter, setup_logging, get_logger


def test_json_formatter() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="hive.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message with %s",
        args=("parameter",),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "hive.test"
    assert parsed["message"] == "Test message with parameter"
    assert "timestamp" in parsed


def test_json_formatter_extra_fields() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="hive.test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=20,
        msg="Node event",
        args=(),
        exc_info=None,
    )
    record.node_id = "node-123"
    record.round_id = 4
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["node_id"] == "node-123"
    assert parsed["round_id"] == 4


def test_text_formatter() -> None:
    formatter = TextFormatter()
    record = logging.LogRecord(
        name="hive.test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=15,
        msg="Something failed",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "[ERROR  ] [hive.test] Something failed" in formatted


def test_setup_logging_and_get_logger() -> None:
    setup_logging(level="DEBUG", format_type="json")
    logger = get_logger("hive.core.test")
    assert logger.name == "hive.core.test"
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) >= 1