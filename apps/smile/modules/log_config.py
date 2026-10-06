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
        short = record.name.rsplit('.', 1)[-1]
        record.short_name = short.ljust(16, '.')
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

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "output" / "logs"
CRASH_LOG = LOG_DIR / "crashes.log"     # one line per unhandled exception, across all starts and apps
CRASH_LINE_LENGTH: int = 200
_app_name: str = "smile"                # the name setup_logging was given: which app crashed


def get_log_buffer() -> LogRingBuffer | None:
    """Return the global log ring buffer (available after setup_logging)."""
    return _log_buffer


def setup_logging(verbose: bool = False, name: str = "smile") -> Path:
    """Configure root logger with console + file handlers.

    Args:
        verbose: If True, console shows DEBUG level; otherwise INFO.
        name: Prefix of the log file in output/logs/ (smile, player1, ...).

    Returns:
        Path to the log file.
    """
    global _log_buffer, _app_name

    _app_name = name
    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(ShortNameFormatter("%(levelname)-7s %(short_name)s %(message)s"))

    # File handler — flush every record so logs survive crashes
    file_handler = FlushFileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ShortNameFormatter("%(asctime)s %(levelname)-7s %(short_name)s %(message)s"))

    # Ring buffer — captures all records for the NiceGUI log drawer
    _log_buffer = LogRingBuffer(maxlen=1000)
    _log_buffer.setLevel(logging.DEBUG)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(_log_buffer)

    return log_file


def record_crash(thread: str, exc_type, exc_value, path: Path | None = None) -> None:
    """One line in the crash log: when, which app, which thread, what. Never raises: it runs in the
    exception hooks."""
    try:
        path = path or CRASH_LOG
        what: str = f"{exc_type.__name__}: {exc_value}".splitlines()[0][:CRASH_LINE_LENGTH]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {_app_name} {thread} {what}\n")
    except Exception:
        pass


def read_crashes(path: Path | None = None) -> tuple[int, str | None, str | None]:
    """(number of crashes, the date of the first, the last line) from the crash log; (0, None, None) without one."""
    path = path or CRASH_LOG
    try:
        lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return 0, None, None
    if not lines:
        return 0, None, None
    return len(lines), lines[0][:10], lines[-1]


def crash_summary(path: Path | None = None) -> str:
    """The first line of every log: what the crash log holds since it was made (delete the file to
    start over)."""
    path = path or CRASH_LOG
    count, since, last = read_crashes(path)
    if not count:
        return "No crashes recorded"
    try:
        shown = path.relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return f"{count} crash{'es' if count != 1 else ''} since {since} ({shown}), last {last}"


def install_excepthooks() -> None:
    """Route unhandled exceptions through the logging system: those of the threads and of the main
    thread. Under pythonw there is no console, so the log file is the only place they show. Each one
    is also a line in the crash log (record_crash), which every start reports (crash_summary)."""
    _logger = logging.getLogger('threading')

    def _main_hook(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        logging.getLogger('main').critical("Unhandled exception in the main thread",
                                           exc_info=(exc_type, exc_value, exc_traceback))
        record_crash('main', exc_type, exc_value)

    sys.excepthook = _main_hook

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name: str = args.thread.name if args.thread else '?'
        _logger.critical(
            "Unhandled exception in thread '%s'",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
            if args.exc_value is not None
            else True,
        )
        record_crash(name, args.exc_type, args.exc_value)

    threading.excepthook = _hook
