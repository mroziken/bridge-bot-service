import logging
import os
from typing import Any


DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] "
    "[request_id=%(request_id)s] %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_REQUEST_ID = "-"
_CONFIGURED = False


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = DEFAULT_REQUEST_ID
        return True


def _parse_log_level(value: str) -> int:
    level_name = value.strip().upper()
    level = logging.getLevelName(level_name)
    if isinstance(level, int):
        return level
    return logging.INFO


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_log_level() -> int:
    env_value = os.getenv("BRIDGE_BOT_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
    return _parse_log_level(env_value)


def payload_logging_enabled() -> bool:
    return _env_flag("BRIDGE_BOT_LOG_PAYLOADS", default=False)


def configure_logging() -> None:
    global _CONFIGURED

    package_logger = logging.getLogger("bridge_bot_service")
    level = get_log_level()

    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT))
        handler.addFilter(RequestContextFilter())
        package_logger.addHandler(handler)
        package_logger.propagate = False
        _CONFIGURED = True
    else:
        for handler in package_logger.handlers:
            handler.addFilter(RequestContextFilter())

    package_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def truncate_for_log(value: Any, limit: int = 500) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"
