import logging
import sys
from datetime import datetime
from pathlib import Path

from .crash_log import LOG_DIR, set_app_name
from .ring_buffer import LogRingBuffer, set_log_buffer, short_name


class ShortNameFormatter(logging.Formatter):
    """A formatter with the logger's short name, padded, as `short_name`."""

    def format(self, record: logging.LogRecord) -> str:
        record.short_name = short_name(record.name).ljust(16, '.')
        return super().format(record)


class FlushFileHandler(logging.FileHandler):
    """A file handler that flushes after every record."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logging(name: str, verbose: bool = False) -> Path:
    """The root logger's console, log file and log buffer; *name* is the app's, the prefix of its log file in logs/.
    Returns the log file's path."""
    set_app_name(name)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{name}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(ShortNameFormatter("%(levelname)-7s %(short_name)s %(message)s"))

    file_handler = FlushFileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ShortNameFormatter("%(asctime)s %(levelname)-7s %(short_name)s %(message)s"))

    log_buffer = LogRingBuffer(maxlen=1000)
    log_buffer.setLevel(logging.DEBUG)
    set_log_buffer(log_buffer)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(log_buffer)
    return log_file
