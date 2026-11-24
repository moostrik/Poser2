import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "logs"
CRASH_LOG = LOG_DIR / "crashes.log"     # one line per unhandled exception, across all starts and apps
CRASH_LINE_LENGTH: int = 200
_app_name: str = "poser"                # the app whose crashes are written


def set_app_name(name: str) -> None:
    """Store the name the crash log's lines give the app."""
    global _app_name
    _app_name = name


def record_crash(thread: str, exc_type: type[BaseException], exc_value: BaseException | None,
                 path: Path | None = None) -> None:
    """One line in the crash log: when, which app, which thread, what."""
    try:
        path = path or CRASH_LOG
        what: str = f"{exc_type.__name__}: {exc_value}".splitlines()[0][:CRASH_LINE_LENGTH]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {_app_name} {thread} {what}\n")
    except Exception:
        logger.exception("Could not write the crash log %s", path)


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
    """The crash log in one line: the number of crashes, the first one's date, the log's path and its last line."""
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
    """The exception hooks of the main thread and the other threads: each unhandled exception is logged and is a
    line in the crash log."""
    thread_logger = logging.getLogger('threading')

    def _main_hook(exc_type: type[BaseException], exc_value: BaseException, exc_traceback) -> None:
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
        thread_logger.critical(
            "Unhandled exception in thread '%s'",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
            if args.exc_value is not None
            else True,
        )
        record_crash(name, args.exc_type, args.exc_value)

    threading.excepthook = _hook
