"""loguru logging utilities."""

import datetime
import functools
import inspect
import logging
import os
import sys
import time
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any

from loguru import logger

# Lazy initialization flag
_logging_initialized = False
_logger_instance = None
_colors_enabled = True  # Track color state separately


class Rotator:
    """Rotates log files based on size and time constraints.

    This class determines when a log file should be rotated, either when
    it exceeds a specified size or a specified time threshold.

    Author: YOGA, u
    """

    def __init__(self, *, size: int, at: datetime.time):
        """Initialize the Rotator with a size limit and a time threshold.

        Args:
            size (int): Maximum allowed log file size in bytes before rotation.
            at (datetime.time): Time of day when rotation should occur.
        """
        now = datetime.datetime.now()
        self._size_limit = size
        self._time_limit = now.replace(hour=at.hour, minute=at.minute, second=at.second)
        if now >= self._time_limit:
            self._time_limit += datetime.timedelta(days=1)

    def should_rotate(self, message: Any, file: Any) -> bool:
        """Determine whether the log file should be rotated.

        Args:
            message (Any): The log message to be written.
            file (Any): The file object representing the log file.

        Returns:
            bool: True if the file should be rotated, False otherwise.
        """
        file.seek(0, 2)
        if file.tell() + len(message) > self._size_limit:
            return True
        excess = message.record["time"].timestamp() - self._time_limit.timestamp()
        if excess >= 0:
            elapsed_days = datetime.timedelta(seconds=excess).days
            self._time_limit += datetime.timedelta(days=elapsed_days + 1)
            return True
        return False


# Global rotator & opener for log file rotation and permissions
rotator = Rotator(size=10_000_000, at=datetime.time(0, 0, 0))


def opener(file: str, flags: int) -> int:
    """Set log file permission to 600 (rw-------) for security.

    Args:
        file (str): The path to the log file.
        flags (int): Flags for opening the file.

    Returns:
        int: File descriptor.
    """
    return os.open(file, flags, 0o600)


def get_colored_logger(colors: bool = True):
    r"""Get a logger instance with optional colorization.

    This function provides a lazy-initialized logger with colorization
    options that can be preserved across the module.

    Args:
        colors (bool): Enable color formatting. Defaults to True.

    Returns:
        Logger instance with colorization settings

    Example:
        >>> # Get colored logger for the whole module
        >>> logger = get_colored_logger(colors=True)
        >>> logger.info("This <green>works</>!")
        >>>
        >>> # Preserve colors for chained opt() calls
        >>> logger.opt(raw=True).info("It <green>still</> works!\\n")
    """
    global _logger_instance, _colors_enabled

    if _logger_instance is None or _colors_enabled != colors:
        # Lazy initialization with color support
        colored_logger = logger.opt(colors=colors)
        # Preserve colors for subsequent opt() calls
        colored_logger.opt = partial(colored_logger.opt, colors=colors)
        _colors_enabled = colors
        _logger_instance = colored_logger

    return _logger_instance


def setup_logging(
    log_level: str = "DEBUG",
    diagnose: bool = True,
    log_path: str = "logs",
    serialize: bool = True,
    colorize: bool = True,
    lazy_init: bool = True,
):
    """Configure and initialize Loguru logging for the application.

    This function sets up Loguru logging with custom rotation, retention,
    and formatting for both console and file outputs. It also intercepts
    standard logging to route through Loguru, ensuring consistent logging
    behavior across the application.

    Args:
        log_level (str, optional): The minimum log level for console output.
            Defaults to "DEBUG". Can be overridden with LOGURU_LEVEL env var.
        diagnose (bool, optional): Whether to enable Loguru's diagnose mode
            for detailed exception tracebacks. Defaults to True.
            Can be overridden with LOGURU_DIAGNOSE env var.
        log_path (str, optional): Directory path where log files will be stored.
            Defaults to "logs".
        serialize (bool, optional): Whether to serialize logs as JSON.
            True for structured JSON logs, False for human-readable text.
            Defaults to True.
        colorize (bool, optional): Whether to enable colorized output for console.
            Defaults to True.
        lazy_init (bool, optional): Whether to use lazy initialization for better
            performance. Defaults to True.

    Example:
        >>> from shared.mylog import setup_logging
        >>> # Basic setup with colors
        >>> setup_logging(log_level="INFO", colorize=True)
        >>>
        >>> # Production setup with text logs, no colors
        >>> setup_logging(
        ...     log_level="WARNING",
        ...     diagnose=False,
        ...     serialize=False,
        ...     colorize=False,
        ... )
        >>>
        >>> # Development setup with detailed debugging and colors
        >>> setup_logging(
        ...     log_level="DEBUG",
        ...     diagnose=True,
        ...     serialize=True,
        ...     colorize=True,
        ... )

    Environment Variables:
        LOGURU_LEVEL: Override log_level parameter
        LOGURU_DIAGNOSE: Override diagnose parameter (true/false)

    Author:
        YOGA (u)
    """
    global _logging_initialized, _logger_instance

    # Lazy initialization check
    if lazy_init and _logging_initialized:
        logger.debug("Logging already initialized, skipping setup")
        return get_colored_logger(colorize)

    # Support environment variable overrides
    log_level = os.getenv("LOGURU_LEVEL", log_level).upper()
    diagnose = os.getenv("LOGURU_DIAGNOSE", str(diagnose)).lower() in (
        "true",
        "1",
        "yes",
    )

    try:
        logger.remove()
        logs_dir = Path(log_path)
        logs_dir.mkdir(exist_ok=True)

        # Enhanced log format with color support
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{process.name}:{thread.name}</cyan> | <blue>{name}:{function}:{line}</blue> | {message}"
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {process.name}:{thread.name} | {name}:{function}:{line} | {message}"  # No colors for file output

        # Console handler with colorization
        logger.add(
            sink=sys.stderr,
            format=log_format,
            level=log_level,
            enqueue=True,
            backtrace=True,
            diagnose=diagnose,
            colorize=colorize,  # Enable/disable colors
        )

        # File handlers (no colorization for files)
        logger.add(
            logs_dir / "app.log",
            rotation=rotator.should_rotate,
            retention="7 days",
            level="INFO",
            format=file_format,
            serialize=serialize,
            enqueue=True,
            backtrace=True,
            diagnose=diagnose,
            opener=opener,
            colorize=False,  # No colors in files
        )
        logger.add(
            logs_dir / "error.log",
            rotation=rotator.should_rotate,
            retention="14 days",
            level="ERROR",
            format=file_format,
            serialize=serialize,
            enqueue=True,
            backtrace=True,
            diagnose=diagnose,
            opener=opener,
            colorize=False,  # No colors in files
        )

        class InterceptHandler(logging.Handler):
            """Custom handler to intercept standard logging and route to Loguru."""

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                # Optimize frame detection for performance
                frame = inspect.currentframe()
                depth = 2
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        # Remove all handlers from root logger to ensure only InterceptHandler is active
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # Mark as initialized for lazy loading
        _logging_initialized = True

        logger.info(
            f"Loguru configured | level={log_level} | diagnose={diagnose} | dir={logs_dir} | colorize={colorize}"
        )

    except Exception as e:
        # Fallback to basic stderr logging if setup fails
        logger.remove()
        logger.add(sys.stderr, level="WARNING", colorize=False)
        logger.error(f"Failed to setup logging: {e}")
        raise
    else:
        return _logger_instance


def setup_library_logging(library_name: str) -> None:
    """Configure logging for library usage.

    Call this function in your library's __init__.py to disable logging by default.
    Users can enable it later with logger.enable(library_name).

    Args:
        library_name: The name of your library (usually __name__.split('.')[0])

    Example:
        >>> # In your library's __init__.py
        >>> from shared.mylog import setup_library_logging
        >>> setup_library_logging(__name__.split(".")[0])
    """
    logger.disable(library_name)
    logger.info(f"Library '{library_name}' logging disabled by default")


def logger_wraps(*, entry: bool = True, exit: bool = True, level: str = "DEBUG"):
    """Decorator to wrap a function with entry and exit logging.

    Args:
        entry (bool, optional): Whether to log function entry. Defaults to True.
        exit (bool, optional): Whether to log function exit. Defaults to True.
        level (str, optional): Log level to use. Defaults to "DEBUG".

    Returns:
        Callable: Decorated function with logging.
    """

    def wrapper(func: Callable) -> Callable:
        name = func.__name__
        doc = func.__doc__ or "No description"

        @functools.wraps(func)
        def wrapped(*args, **kwargs) -> Any:
            # Use colored logger if available
            log_instance = _logger_instance or logger

            if entry:
                log_instance.log(
                    level,
                    f"<cyan>Entering</cyan> '<yellow>{name}</yellow>' (args={len(args)}, kwargs={len(kwargs)}) | Description: <dim>{doc.strip().splitlines()[0]}</dim>",
                )
            result = func(*args, **kwargs)
            if exit:
                log_instance.log(
                    level,
                    f"<cyan>Exiting</cyan> '<yellow>{name}</yellow>' | Result: <green>{result!r}</green>",
                )
            return result

        return wrapped

    return wrapper


def timer(operation: str | None = None):
    """Decorator to log the duration of a function call.

    Args:
        operation (str | None, optional): Name of the operation for logging.
            If None, uses the function name. Defaults to None.

    Returns:
        Callable: Decorated function with timing logs.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation or func.__name__.upper()
            start = time.perf_counter()

            # Use colored logger if available
            log_instance = _logger_instance or logger

            try:
                log_instance.info(f"<blue>[{op_name}]</blue> <dim>Starting...</dim>")
                result = func(*args, **kwargs)
            except Exception:
                duration = time.perf_counter() - start
                log_instance.exception(
                    f"<red>[{op_name}]</red> <red>Failed</red> after <yellow>{duration:.3f}s</yellow>"
                )
                raise
            else:
                duration = time.perf_counter() - start
                log_instance.info(
                    f"<green>[{op_name}]</green> <green>Completed</green> in <yellow>{duration:.3f}s</yellow>"
                )
                return result

        return wrapper

    return decorator


class LogContext:
    """Context manager for logging the duration and outcome of a code block.

    Usage:
        with LogContext("OPERATION_NAME", level="INFO"):
            # code block
    """

    def __init__(self, operation: str, level: str = "INFO"):
        """Initialize the LogContext.

        Args:
            operation (str): Name of the operation for logging.
            level (str, optional): Log level to use. Defaults to "INFO".
        """
        self.operation = operation
        self.level = level
        self.start_time: float | None = None

    def __enter__(self) -> "LogContext":
        """Enter the runtime context and start timing.

        Returns:
            LogContext: The context manager instance.
        """
        self.start_time = time.perf_counter()
        # Use colored logger if available
        log_instance = _logger_instance or logger
        log_instance.log(
            self.level, f"<blue>[{self.operation}]</blue> <dim>Starting...</dim>"
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context and log the result and duration.

        Args:
            exc_type (type[BaseException] | None): Exception type, if any.
            exc_val (BaseException | None): Exception value, if any.
            exc_tb (TracebackType | None): Traceback, if any.
        """
        if self.start_time is None:
            return
        duration = time.perf_counter() - self.start_time

        # Use colored logger if available
        log_instance = _logger_instance or logger

        if exc_type:
            log_instance.error(
                f"<red>[{self.operation}]</red> <red>Failed</red> after <yellow>{duration:.3f}s</yellow>: <red>{exc_val}</red>"
            )
        else:
            log_instance.log(
                self.level,
                f"<green>[{self.operation}]</green> <green>Completed</green> in <yellow>{duration:.3f}s</yellow>",
            )


def log_with_stacktrace(message: str, level: str = "DEBUG") -> None:
    """Log a message with the current stacktrace (for debugging purposes).

    This utility logs a message and appends the current Python stacktrace, even if no exception is raised.
    Useful for tracing code flow or debugging complex call chains.

    Args:
        message (str): The message to log.
        level (str, optional): Log level to use (e.g., "DEBUG", "INFO"). Defaults to "DEBUG".

    Example:
        >>> from shared.mylog import log_with_stacktrace
        >>> log_with_stacktrace("Checkpoint reached", level="INFO")

    Note:
        - Only use in development or debugging sessions. Do not use in production for every log (very verbose).
        - Does not require changing the main logger handler or format.
    """
    stack = "".join(traceback.format_stack())
    logger.log(level, f"{message}\nStacktrace:\n{stack}")


# Performance optimization: Lazy logger instance
def get_logger():
    """Get the configured logger instance with lazy initialization.

    Returns:
        Configured logger instance with performance optimizations.
    """
    global _logger_instance
    if _logger_instance is None:
        # Initialize with default settings if not already done
        setup_logging(lazy_init=True)
    return _logger_instance


__all__ = [
    "LogContext",
    "get_colored_logger",
    "get_logger",
    "log_with_stacktrace",
    "logger",
    "logger_wraps",
    "setup_library_logging",
    "setup_logging",
    "timer",
]


# Example Of Usage:
# from shared.mylog import setup_logging, get_colored_logger, logger

# # Setup with colors enabled
# setup_logging(log_level="INFO", colorize=True)

# # Get colored logger for module-wide use
# logger = get_colored_logger(colors=True)
# logger.info("This <green>works</>!")
# logger.opt(raw=True).info("It <green>still</> works!\n")

# # Production setup without colors
# setup_logging(
#     log_level="WARNING",
#     diagnose=False,
#     serialize=False,
#     colorize=False,  # Disable colors for production
#     log_path="production_logs"
# )

# # Development setup with colors and lazy init
# setup_logging(
#     log_level="DEBUG",
#     diagnose=True,
#     serialize=True,
#     colorize=True,  # Enable colors for development
#     lazy_init=True  # Performance optimization
# )

# # Using colored decorators
# @logger_wraps(level="INFO")  # Now with colors!
# @timer("DATABASE_QUERY")     # Now with colors!
# def query_database():
#     # Your code here
#     pass

# # Colored context manager
# with LogContext("FILE_PROCESSING", level="INFO"):  # Now with colors!
#     # Your code here
#     pass
# with LogContext("FILE_PROCESSING", level="INFO"):  # Now with colors!
#     # Your code here
#     pass
