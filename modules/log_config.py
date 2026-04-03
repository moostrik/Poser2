import logging
import sys
from datetime import datetime
from pathlib import Path


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


def setup_logging(verbose: bool = False) -> Path:
    """Configure root logger with console + file handlers.

    Args:
        verbose: If True, console shows DEBUG level; otherwise INFO.

    Returns:
        Path to the log file.
    """
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

    root.addHandler(console)
    root.addHandler(file_handler)

    return log_file
