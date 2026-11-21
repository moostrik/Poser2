import logging
import traceback
from collections import deque
from datetime import datetime
from itertools import islice
from threading import Lock
from typing import NamedTuple


class LogEntry(NamedTuple):
    """One log record as the panel's log drawer shows it."""
    timestamp: str
    level: str
    source: str
    message: str


def short_name(name: str) -> str:
    """A logger's name without its packages: "modules.camera.sync" is "sync"."""
    return name.rsplit('.', 1)[-1]


class LogRingBuffer(logging.Handler):
    """The latest log entries, bounded, read from a cursor."""

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__()
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._counter: int = 0  # total entries ever added

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.add(LogEntry(
                timestamp=self.format_time(record),
                level=record.levelname,
                source=short_name(record.name),
                message=self.format_message(record),
            ))
        except Exception:
            self.handleError(record)

    def add(self, entry: LogEntry) -> None:
        """Append an entry made elsewhere."""
        with self._lock:
            self._entries.append(entry)
            self._counter += 1

    @property
    def cursor(self) -> int:
        """The cursor past every entry so far."""
        with self._lock:
            return self._counter

    def get_entries_since(self, cursor: int) -> tuple[int, list[LogEntry]]:
        """(the new cursor, the entries added since *cursor*); cursor 0 is everything buffered."""
        with self._lock:
            total = self._counter
            oldest_cursor = total - len(self._entries)
            skip = max(cursor, oldest_cursor) - oldest_cursor
            return total, list(islice(self._entries, skip, None))

    @staticmethod
    def format_time(record: logging.LogRecord) -> str:
        return datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

    @staticmethod
    def format_message(record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1]:
            msg += '\n' + ''.join(traceback.format_exception(*record.exc_info)).rstrip()
        return msg


_log_buffer: LogRingBuffer | None = None


def set_log_buffer(buffer: LogRingBuffer) -> None:
    """Store the app's log buffer."""
    global _log_buffer
    _log_buffer = buffer


def get_log_buffer() -> LogRingBuffer | None:
    """The app's log buffer; None before setup_logging."""
    return _log_buffer
