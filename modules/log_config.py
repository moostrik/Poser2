import logging
import sys
import threading
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path
from threading import Lock


LogEntry = namedtuple('LogEntry', ['timestamp', 'level', 'source', 'message'])


class ShortNameFormatter(logging.Formatter):
    """Formatter that shortens logger names to last dotted segment."""

    def format(self, record):
        # "modules.oak.camera.camera" → "camera"
        record.short_name = record.name.rsplit('.', 1)[-1]
        return super().format(record)


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record for crash safety."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class LogRingBuffer(logging.Handler):
    """Handler that stores recent log records in a bounded ring buffer.

    Thread-safe. Consumers poll via get_entries_since(cursor) to retrieve
    only new entries since their last read.
    """

    def __init__(self, maxlen: int = 1000):
        super().__init__()
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._counter: int = 0  # total records ever appended

    def emit(self, record):
        try:
            short_name = record.name.rsplit('.', 1)[-1]
            entry = LogEntry(
                timestamp=self.format_time(record),
                level=record.levelname,
                source=short_name,
                message=self.format_message(record),
            )
            with self._lock:
                self._entries.append(entry)
                self._counter += 1
        except Exception:
            self.handleError(record)

    def get_entries_since(self, cursor: int) -> tuple[int, list[LogEntry]]:
        """Return (new_cursor, entries) added since *cursor*.

        First call: pass cursor=0 to get everything currently buffered.
        """
        with self._lock:
            total = self._counter
            buffered = len(self._entries)
            oldest_cursor = total - buffered
            if cursor < oldest_cursor:
                cursor = oldest_cursor
            skip = cursor - oldest_cursor
            return total, list(self._entries)[skip:]

    @staticmethod
    def format_time(record) -> str:
        return datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

    @staticmethod
    def format_message(record) -> str:
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1]:
            import traceback
            msg += '\n' + ''.join(traceback.format_exception(*record.exc_info)).rstrip()
        return msg


_log_buffer: LogRingBuffer | None = None


def get_log_buffer() -> LogRingBuffer | None:
    """Return the global log ring buffer (available after setup_logging)."""
    return _log_buffer


def setup_logging(verbose: bool = False) -> Path:
    """Configure root logger with console + file handlers.

    Args:
        verbose: If True, console shows DEBUG level; otherwise INFO.

    Returns:
        Path to the log file.
    """
    global _log_buffer

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"poser_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(ShortNameFormatter("%(levelname)-7s %(short_name)s: %(message)s"))

    # File handler — flush every record so logs survive crashes
    file_handler = FlushFileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ShortNameFormatter("%(asctime)s %(levelname)-7s %(short_name)s: %(message)s"))

    # Ring buffer — captures all records for the NiceGUI log drawer
    _log_buffer = LogRingBuffer(maxlen=1000)
    _log_buffer.setLevel(logging.DEBUG)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(_log_buffer)

    return log_file


def install_thread_excepthook() -> None:
    """Route unhandled thread exceptions through the logging system."""
    _logger = logging.getLogger('threading')

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        _logger.critical(
            "Unhandled exception in thread '%s'",
            args.thread.name if args.thread else '?',
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
            if args.exc_value is not None
            else True,
        )

    threading.excepthook = _hook
