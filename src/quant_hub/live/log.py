"""Structured JSON logging: one line per event, appended to a session log file too."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        body = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            body.update(extra)
        if record.exc_info:
            body["exc"] = self.formatException(record.exc_info)
        return json.dumps(body, sort_keys=True, default=str)


def get_logger(
    name: str, file: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Logger writing JSON lines to stderr and optionally to a file (append-only)."""
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
    if file is not None and not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file)
        fh.setFormatter(JsonFormatter())
        log.addHandler(fh)
    return log


def event(log: logging.Logger, msg: str, level: int = logging.INFO, **data) -> None:
    log.log(level, msg, extra={"data": data})
