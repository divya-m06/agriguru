import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ml",
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger that logs both to standard output and optionally to a file.

    Args:
        name: Name of the logger.
        log_file: Optional path to a log file.
        level: Logging level (e.g. logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is invoked repeatedly
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
