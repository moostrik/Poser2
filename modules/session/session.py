"""Session — recording lifecycle coordinator.

Owns the chunk timer.  When ``settings.record`` goes True the timer starts;
at each ``chunk_length`` boundary it fires the shared ``split`` button which
propagates to all child recorders via the settings share mechanism.
"""
import threading
import time

from .settings import SessionSettings

import logging
logger = logging.getLogger(__name__)


class Session:

    def __init__(self, settings: SessionSettings) -> None:
        self.settings = settings
        self._timer_thread: threading.Thread | None = None
        self._timer_stop = threading.Event()

        settings.bind(SessionSettings.record, self._on_record)

    def _on_record(self, value: bool) -> None:
        if value:
            self._start_timer()
        else:
            self._stop_timer()

    # ── Chunk timer ──────────────────────────────────────────────────────

    def _start_timer(self) -> None:
        self._stop_timer()
        self._timer_stop.clear()
        self._timer_thread = threading.Thread(
            target=self._timer_loop, daemon=True, name="SessionChunkTimer"
        )
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        if self._timer_thread is not None:
            self._timer_stop.set()
            self._timer_thread.join()
            self._timer_thread = None

    def _timer_loop(self) -> None:
        chunk_start = time.time()
        while not self._timer_stop.is_set():
            chunk_length = self.settings.split_seconds
            if chunk_length <= 0:
                # No chunking — just idle until stopped
                self._timer_stop.wait(0.1)
                continue

            elapsed = time.time() - chunk_start
            remaining = chunk_length - elapsed
            if remaining > 0:
                self._timer_stop.wait(remaining)
                continue

            # Time to split
            SessionSettings.split.fire(self.settings)
            chunk_start += chunk_length
