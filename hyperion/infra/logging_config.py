"""
Hyperion Logging System
=======================
Structured JSON logging with rotating file handlers.
Supports: console, file, remote (future: ELK).
"""
from __future__ import annotations

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """JSON-structured log formatter (ELK-compatible)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": getattr(record, "module", ""),
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console output for readability."""

    COLORS = {
        "DEBUG": "\033[36m",   # Cyan
        "INFO": "\033[32m",    # Green
        "WARNING": "\033[33m", # Yellow
        "ERROR": "\033[31m",   # Red
        "CRITICAL": "\033[35m",# Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname_colored = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = "hyperion",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    json_format: bool = False,
    console: bool = True,
) -> logging.Logger:
    """Set up structured logging for Hyperion.

    Args:
        name: Logger name
        level: Log level
        log_dir: Directory for rotating file logs
        json_format: Use JSON format for files
        console: Enable console output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        fmt = "%(asctime)s | %(levelname_colored)-8s | %(name)s:%(lineno)d | %(message)s"
        ch.setFormatter(ColoredFormatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(ch)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
        )
        fh.setLevel(level)
        if json_format:
            fh.setFormatter(JsonFormatter())
        else:
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
            ))
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "hyperion") -> logging.Logger:
    """Get or create a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logging(name)
    return logger