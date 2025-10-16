"""Logging utilities for the Rosetta Transformer project.

This module provides functions for setting up and managing logging across
the application with consistent formatting and output handling.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Detailed logging format with module and function information
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "[%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s"
)


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
    detailed: bool = False,
) -> None:
    """Set up logging configuration for the application.

    This function configures the root logger with consistent formatting
    and output handlers. It can log to both console and file simultaneously.

    Args:
        log_file: Optional path to log file. If provided, logs will be written
                 to both console and file.
        level: Logging level as string. Valid values: DEBUG, INFO, WARNING,
              ERROR, CRITICAL. Default is INFO.
        format_string: Custom format string for log messages. If None, uses
                      DEFAULT_FORMAT or DETAILED_FORMAT based on `detailed`.
        detailed: If True and format_string is None, uses DETAILED_FORMAT
                 which includes file, line number, and function information.

    Raises:
        ValueError: If an invalid logging level is provided

    Example:
        >>> setup_logging(log_file="training.log", level="DEBUG")
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    # Validate and get logging level
    level_upper = level.upper()
    numeric_level = getattr(logging, level_upper, None)

    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level: {level}. "
            "Valid options: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )

    # Determine format string
    if format_string is None:
        format_string = DETAILED_FORMAT if detailed else DEFAULT_FORMAT

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress overly verbose third-party loggers
    _suppress_noisy_loggers()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    This function returns a logger instance that inherits configuration
    from the root logger set up by setup_logging().

    Args:
        name: Name for the logger, typically __name__ of the calling module

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
        >>> logger.debug("Debug information")
    """
    return logging.getLogger(name)


def _suppress_noisy_loggers() -> None:
    """Suppress overly verbose logging from third-party libraries.

    This internal function reduces log spam from common libraries that
    tend to be verbose at INFO level.
    """
    # Set third-party loggers to WARNING level
    noisy_loggers = [
        "transformers",
        "datasets",
        "urllib3",
        "requests",
        "mlflow",
        "wandb",
        "PIL",
        "matplotlib",
        "numba",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def add_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.FileHandler:
    """Add a file handler to an existing logger.

    This function allows adding additional file outputs to a specific logger
    without affecting the global logging configuration.

    Args:
        logger: Logger instance to add the file handler to
        log_file: Path to the log file
        level: Optional logging level for this handler. If None, uses the
              logger's level.
        format_string: Optional format string for this handler. If None,
                      uses DEFAULT_FORMAT.

    Returns:
        logging.FileHandler: The created file handler

    Example:
        >>> logger = get_logger(__name__)
        >>> handler = add_file_handler(logger, "model_training.log", "DEBUG")
        >>> logger.debug("This goes to both console and file")
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")

    # Set level
    if level is not None:
        level_upper = level.upper()
        numeric_level = getattr(logging, level_upper, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        file_handler.setLevel(numeric_level)

    # Set formatter
    if format_string is None:
        format_string = DEFAULT_FORMAT
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return file_handler


def log_dict(logger: logging.Logger, data: dict, level: str = "INFO") -> None:
    """Log a dictionary in a formatted, readable way.

    Args:
        logger: Logger instance to use
        data: Dictionary to log
        level: Logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> logger = get_logger(__name__)
        >>> config = {"model": "bert-base", "batch_size": 32}
        >>> log_dict(logger, config, "INFO")
    """
    level_upper = level.upper()
    log_func = getattr(logger, level_upper.lower())

    log_func("Dictionary contents:")
    for key, value in data.items():
        log_func(f"  {key}: {value}")


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars.

    This handler ensures that log messages don't interfere with tqdm
    progress bars by writing through tqdm's write method.

    Example:
        >>> from tqdm import tqdm
        >>> logger = get_logger(__name__)
        >>> logger.addHandler(TqdmLoggingHandler())
        >>> for i in tqdm(range(100)):
        ...     logger.info(f"Processing {i}")  # Won't break progress bar
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record through tqdm.write().

        Args:
            record: Log record to emit
        """
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # Fall back to standard output if tqdm is not available
            print(self.format(record))
        except Exception:
            self.handleError(record)
