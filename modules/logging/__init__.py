"""The apps' logging: console, log file, the log buffer of the panel's drawer, the crash log."""

from .crash_log import crash_summary, install_excepthooks
from .ring_buffer import LogEntry, LogRingBuffer, get_log_buffer
from .setup import setup_logging
