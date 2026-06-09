from __future__ import annotations

import logging


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging once with a consistent format."""

    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
