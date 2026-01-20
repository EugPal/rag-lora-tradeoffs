from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


def setup_logging(name: str | None = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        message = f"{label} took {duration:.3f}s"
        if logger:
            logger.info(message)
        else:
            print(message)
