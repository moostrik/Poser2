"""Session — recording lifecycle coordinator.

Owns the chunk timer.  When ``start`` is pressed the timer starts;
at each ``split_seconds`` boundary it fires the shared ``split`` button which
propagates to all child recorders via the settings share mechanism.
``stop`` halts the timer.  The ``running`` field reflects current state.
"""
import threading
import time

from modules.settings import BaseSettings, Field, Widget

import logging
logger = logging.getLogger(__name__)


class SessionSettings(BaseSettings):

    start:         Field[bool]  = Field(False, widget=Widget.button, description="Start session")
    stop:          Field[bool]  = Field(False, widget=Widget.button, description="Stop session")
    running:       Field[bool]  = Field(False, access=Field.READ, description="Session running")
    output_path:   Field[str]   = Field("recordings", description="Recordings output directory", access=Field.INIT)
    name:          Field[str]   = Field("", widget=Widget.input, description="Recording name")
    split:         Field[bool]  = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds: Field[float] = Field(10, min=1, max=60, widget=Widget.number, description="Split recording into chunks of this length (seconds)")


class Session:

    def __init__(self, settings: SessionSettings) -> None:
        self.settings = settings
        self._timer_thread: threading.Thread | None = None
        self._timer_stop = threading.Event()

        settings.bind(SessionSettings.start, self._on_start)
        settings.bind(SessionSettings.stop, self._on_stop)

    def _on_start(self, _=None) -> None:
        if self.settings.running:
            return
        self.settings.running = True
        self._start_timer()

    def _on_stop(self, _=None) -> None:
        if not self.settings.running:
            return
        self._stop_timer()
        self.settings.running = False

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
