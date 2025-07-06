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
from pathlib import Path
from types import TracebackType
from typing import Any

from loguru import logger


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


def setup_logging(
    log_level: str = "DEBUG",
    diagnose: bool = True,
    log_path: str = "logs",
    serialize: bool = True,
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

    Example:
        >>> from shared.mylog import setup_logging
        >>> # Basic setup
        >>> setup_logging(log_level="INFO")
        >>>
        >>> # Production setup with text logs
        >>> setup_logging(
        ...     log_level="WARNING", diagnose=False, serialize=False
        ... )
        >>>
        >>> # Development setup with detailed debugging
        >>> setup_logging(
        ...     log_level="DEBUG", diagnose=True, serialize=True
        ... )

    Environment Variables:
        LOGURU_LEVEL: Override log_level parameter
        LOGURU_DIAGNOSE: Override diagnose parameter (true/false)

    Author:
        YOGA (u)
    """
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

        log_format = "{time:YYYY-MM-DD HH:mm:ss} {level} {process.name}:{thread.name} {name}:{function}:{line} {message}"
        file_format = log_format  # Samakan format file dan terminal
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            enqueue=True,
            backtrace=True,
            diagnose=diagnose,
        )
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
        )

        class InterceptHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                frame, depth = inspect.currentframe(), 2
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
        logger.info(
            f"Loguru configured | level={log_level} | diagnose={diagnose} | dir={logs_dir}"
        )

    except Exception as e:
        # Fallback to basic stderr logging if setup fails
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        logger.error(f"Failed to setup logging: {e}")
        raise


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
            if entry:
                logger.log(
                    level,
                    f"Entering '{name}' (args={len(args)}, kwargs={len(kwargs)}) | Description: {doc.strip().splitlines()[0]}",
                )
            result = func(*args, **kwargs)
            if exit:
                logger.log(level, f"Exiting '{name}' | Result: {result!r}")
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
            try:
                logger.info(f"[{op_name}] Starting...")
                result = func(*args, **kwargs)
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"[{op_name}] Failed after {duration:.3f}s: {e}")
                raise
            else:
                duration = time.perf_counter() - start
                logger.info(f"[{op_name}] Completed in {duration:.3f}s")
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
        logger.log(self.level, f"[{self.operation}] Starting...")
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
        if exc_type:
            logger.error(f"[{self.operation}] Failed after {duration:.3f}s: {exc_val}")
        else:
            logger.log(self.level, f"[{self.operation}] Completed in {duration:.3f}s")


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


__all__ = [
    "LogContext",
    "log_with_stacktrace",
    "logger",
    "logger_wraps",
    "setup_library_logging",
    "setup_logging",
    "timer",
]


# Example Of Usage:
# from shared.mylog import setup_logging, logger

# # Setup dasar
# setup_logging(log_level="INFO")

# # Production setup
# setup_logging(
#     log_level="WARNING",
#     diagnose=False,
#     serialize=False,  # Human-readable logs
#     log_path="production_logs"
# )

# # Development setup
# setup_logging(
#     log_level="DEBUG",
#     diagnose=True,
#     serialize=True  # Structured JSON logs
# )
# # Di __init__.py library Anda
# from shared.mylog import setup_library_logging
# setup_library_logging(__name__.split('.')[0])

# User dapat mengaktifkan dengan:
# logger.enable("nama_library")
# from shared.mylog import logger_wraps, timer, LogContext

# @logger_wraps(level="INFO")
# @timer("DATABASE_QUERY")
# def query_database():
#     # Your code here
#     pass

# # Context manager
# with LogContext("FILE_PROCESSING", level="INFO"):
#     # Your code here
#     pass
