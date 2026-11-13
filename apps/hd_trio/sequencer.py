"""Sequencer — HD Trio's stage-based show sequencer with settings-driven control, ticked by the render loop."""

from threading import Lock
from typing import Callable

from modules.settings import BaseSettings, Field, Widget
from modules.session import SequencerState

import logging
logger = logging.getLogger(__name__)


class SequencerSettings(BaseSettings):
    """Base settings for Sequencer.

    Subclass this per project and override:
    - ``stages`` with your stage enum list (the playlist of stages to play)
    - ``durations`` with your per-stage durations
    - ``stage`` with your stage enum type (read-only output)

    Example::

        class ShowSequencerSettings(SequencerSettings):
            stages:    Field[list[ShowStage]] = Field(list(ShowStage), widget=Widget.playlist)
            durations: Field[list[float]]     = Field([3.0, 10.0, 5.0])
            stage:     Field[ShowStage]       = Field(ShowStage.START, access=Field.READ)
    """
    stages:         Field[list[int]]   = Field([0], description="Stages to play")
    durations:      Field[list[float]] = Field([0.0], min=0.0, description="Duration per stage (seconds)")
    stage:          Field[int]         = Field(0, access=Field.READ, description="Current stage")
    stage_progress: Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Stage progress")
    progress:       Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Overall progress")
    start:          Field[bool]        = Field(False, widget=Widget.button, description="Start show", newline=True)
    stop:           Field[bool]        = Field(False, widget=Widget.button, description="Stop show")
    running:        Field[bool]        = Field(False, access=Field.READ, description="Show running")
    skip:           Field[bool]        = Field(False, widget=Widget.button, description="Skip to next stage")
    loop:           Field[bool]        = Field(False, description="Loop show back to start")


class Sequencer:
    """Tick-based sequencer that progresses through a playlist of stages.

    The ``stages`` list defines which stages to play and in what order.
    Each entry indexes into ``durations`` for that stage's length.
    Call ``update(now)`` every frame (the renderer's update callback hands it the frame time). The panel's start,
    stop, skip and playlist changes are requests, taken and carried out at the next tick, so the show changes on
    the tick's thread only and its stages start at the tick's time; the state callbacks run there.

    Example::

        settings = ShowSequencerSettings()
        sequencer = Sequencer(settings)
        sequencer.add_state_callback(on_state)
        renderer.add_update_callback(sequencer.update)

        settings.stages = [0, 1, 2, 3, 4]   # Full show
        SequencerSettings.start.fire(settings)

        settings.stages = [2]                # Test single stage
        SequencerSettings.start.fire(settings)
    """

    def __init__(self, settings: SequencerSettings) -> None:
        self.settings = settings

        if not settings.durations:
            raise ValueError("settings.durations must contain at least one entry")

        self._state_callbacks: list[Callable[[SequencerState], None]] = []      # run in registration order
        self._callback_lock: Lock = Lock()

        # The panel's requests, set on its thread, taken at the next tick
        self._request_lock: Lock = Lock()
        self._start_requested: bool = False
        self._stop_requested: bool = False
        self._skip_requested: bool = False
        self._playlist_requested: bool = False

        self._active = False
        self._pos: int = 0
        self._stage_start: float = 0.0

        # Cached playlist data (rebuilt in _start_show)
        self._playlist: list[int] = []
        self._stage_durations: list[float] = []
        self._cumulative: list[float] = []
        self._total_duration: float = 0.0

        settings.bind(SequencerSettings.start, self._on_start)
        settings.bind(SequencerSettings.stop, self._on_stop)
        settings.bind(SequencerSettings.skip, self._on_skip)
        settings.bind(SequencerSettings.durations, self._on_playlist_changed)
        settings.bind(SequencerSettings.stages, self._on_playlist_changed)
        self._set_idle_state()

    def stop(self) -> None:
        """Teardown: the settings bindings go."""
        self.settings.unbind(SequencerSettings.start, self._on_start)
        self.settings.unbind(SequencerSettings.stop, self._on_stop)
        self.settings.unbind(SequencerSettings.skip, self._on_skip)
        self.settings.unbind(SequencerSettings.durations, self._on_playlist_changed)
        self.settings.unbind(SequencerSettings.stages, self._on_playlist_changed)

    # -- Settings callbacks: requests for the next tick ----------------------

    def _on_start(self, _=None) -> None:
        with self._request_lock:
            self._start_requested = True

    def _on_stop(self, _=None) -> None:
        with self._request_lock:
            self._stop_requested = True

    def _on_skip(self, value: bool) -> None:
        if value:
            with self._request_lock:
                self._skip_requested = True

    def _on_playlist_changed(self, _=None) -> None:
        with self._request_lock:
            self._playlist_requested = True

    def _take_requests(self, now: float) -> None:
        """The panel's requests since the last tick, carried out at `now`: a stop before a start, then a skip, then a
        playlist change."""
        with self._request_lock:
            start, stop, skip, playlist = (self._start_requested, self._stop_requested, self._skip_requested,
                                           self._playlist_requested)
            self._start_requested = self._stop_requested = self._skip_requested = self._playlist_requested = False
        if stop and self._active:
            self._stop_show()
        if start and not self._active:
            self._start_show(now)
        if skip and self._active:
            self._advance_stage(now)
        if playlist and self._active:
            self._build_playlist()
            if self._pos >= len(self._playlist):
                self._pos = max(0, len(self._playlist) - 1)

    # -- Playlist cache ------------------------------------------------------

    def _build_playlist(self) -> None:
        """Snapshot the playlist and precompute duration data."""
        durations = self.settings.durations
        self._playlist = [int(s) for s in self.settings.stages]
        self._stage_durations = [
            float(durations[s]) if 0 <= s < len(durations) else 0.0
            for s in self._playlist
        ]
        cumulative = []
        total = 0.0
        for d in self._stage_durations:
            cumulative.append(total)
            total += d
        self._cumulative = cumulative
        self._total_duration = total

    # -- Show lifecycle ------------------------------------------------------

    def _set_idle_state(self) -> None:
        """Set outputs to end-of-show: last playlist stage, full progress."""
        self._build_playlist()
        if self._playlist:
            self.settings.stage = self._playlist[-1]
        self.settings.stage_progress = 1.0
        self.settings.progress = 1.0

    def _start_show(self, now: float) -> None:
        self._build_playlist()
        if not self._playlist:
            logger.warning("empty stages playlist, not starting")
            return
        self._active = True
        self.settings.running = True
        self._pos = 0
        self._enter_stage(now)

    def _stop_show(self) -> None:
        self._active = False
        self.settings.running = False
        self._set_idle_state()
        self._notify_state(0.0)

    def _enter_stage(self, now: float) -> None:
        self._stage_start = now
        self.settings.stage_progress = 0.0
        stage = self._playlist[self._pos]
        self.settings.stage = stage
        self._notify_state(0.0)

    def _advance_stage(self, now: float) -> None:
        next_pos = self._pos + 1
        if next_pos < len(self._playlist):
            self._pos = next_pos
            self._enter_stage(now)
        elif self.settings.loop:
            self._pos = 0
            self._enter_stage(now)
        else:
            SequencerSettings.stop.fire(self.settings)      # the shared stop: the session's recorders stop with the show

    # -- Callbacks -----------------------------------------------------------

    def add_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        with self._callback_lock:
            if callback not in self._state_callbacks:
                self._state_callbacks.append(callback)
        callback(SequencerState(
            stage=self.settings.stage,
            stage_progress=self.settings.stage_progress,
            progress=self.settings.progress,
            elapsed=0.0,
            active=self._active,
        ))

    def remove_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        with self._callback_lock:
            if callback in self._state_callbacks:
                self._state_callbacks.remove(callback)

    def _notify_state(self, elapsed: float) -> None:
        state = SequencerState(
            stage=self.settings.stage,
            stage_progress=self.settings.stage_progress,
            progress=self.settings.progress,
            elapsed=elapsed,
            active=self._active,
        )
        with self._callback_lock:
            callbacks = list(self._state_callbacks)
        for callback in callbacks:
            try:
                callback(state)
            except Exception:
                logger.exception("Sequencer state callback error")

    # -- Tick (called every render frame) ------------------------------------

    def update(self, now: float) -> None:
        """Advance sequencer state. Call once per frame with the frame's time (perf_counter)."""
        self._take_requests(now)
        if not self._active:
            return

        elapsed = now - self._stage_start
        duration = self._stage_durations[self._pos]

        # Stage progress
        self.settings.stage_progress = min(elapsed / duration, 1.0) if duration > 0 else 1.0

        # Overall progress
        if self._total_duration > 0:
            self.settings.progress = min(
                (self._cumulative[self._pos] + elapsed) / self._total_duration, 1.0
            )
        else:
            self.settings.progress = 1.0

        # State callbacks
        self._notify_state(elapsed)

        # Auto-advance
        if duration > 0 and elapsed >= duration:
            self._advance_stage(now)
