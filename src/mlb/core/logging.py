from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logger(name: str = "mlb", log_file: Path | None = None) -> logging.Logger:
    """
    Logger to console + optional JSONL file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(_JsonlFormatter())
        logger.addHandler(fh)

    return logger


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)
