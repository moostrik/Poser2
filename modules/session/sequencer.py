"""Sequencer — stage-based sequencer with config-driven control, ticked by the render loop."""

import time
from dataclasses import dataclass
from typing import Callable

from modules.settings import BaseSettings, Field, Widget

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SequencerState:
    """Immutable snapshot of the sequencer's current state."""
    stage: int
    stage_progress: float
    progress: float
    elapsed: float
    active: bool


class SequencerSettings(BaseSettings):
    """Base configuration for Sequencer.

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
    start:          Field[bool]        = Field(False, widget=Widget.button, newline=True, description="Start show")
    stop:           Field[bool]        = Field(False, widget=Widget.button, description="Stop show")
    running:        Field[bool]        = Field(False, access=Field.READ, description="Show running")
    skip:           Field[bool]        = Field(False, widget=Widget.button, description="Skip to next stage")
    stage:          Field[int]         = Field(0, access=Field.READ, description="Current stage")
    stage_progress: Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Stage progress")
    progress:       Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Overall progress")


class Sequencer:
    """Tick-based sequencer that progresses through a playlist of stages.

    The ``stages`` list defines which stages to play and in what order.
    Each entry indexes into ``durations`` for that stage's length.
    Call ``update()`` every frame (e.g. from ``data_hub.notify_update()``).

    Example::

        config = ShowSequencerSettings()
        sequencer = Sequencer(config)
        sequencer.add_state_callback(on_state)
        data_hub.add_update_callback(sequencer.update)

        config.stages = [0, 1, 2, 3, 4]   # Full show
        SequencerSettings.start.fire(config)

        config.stages = [2]                # Test single stage
        SequencerSettings.start.fire(config)
    """

    def __init__(self, config: SequencerSettings) -> None:
        self.config = config

        if not config.durations:
            raise ValueError("config.durations must contain at least one entry")

        self._state_callbacks: set[Callable[[SequencerState], None]] = set()

        self._active = False
        self._pos: int = 0
        self._stage_start: float = 0.0

        # Cached playlist data (rebuilt in _start_show)
        self._playlist: list[int] = []
        self._stage_durations: list[float] = []
        self._cumulative: list[float] = []
        self._total_duration: float = 0.0

        config.bind(SequencerSettings.start, self._on_start)
        config.bind(SequencerSettings.stop, self._on_stop)
        config.bind(SequencerSettings.skip, self._on_skip)
        self._set_idle_state()

    # -- Settings callbacks --------------------------------------------------

    def _on_start(self, _=None) -> None:
        if not self._active:
            self._start_show()

    def _on_stop(self, _=None) -> None:
        if self._active:
            self._stop_show()

    def _on_skip(self, value: bool) -> None:
        if value and self._active:
            self._advance_stage()

    # -- Playlist cache ------------------------------------------------------

    def _build_playlist(self) -> None:
        """Snapshot the playlist and precompute duration data."""
        durations = self.config.durations
        self._playlist = [int(s) for s in self.config.stages]
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
            self.config.stage = self._playlist[-1]
        self.config.stage_progress = 1.0
        self.config.progress = 1.0

    def _start_show(self) -> None:
        self._build_playlist()
        if not self._playlist:
            logger.warning("Sequencer: empty stages playlist, not starting")
            return
        self._active = True
        self.config.running = True
        self._pos = 0
        self._enter_stage()

    def _stop_show(self) -> None:
        self._active = False
        self.config.running = False
        self._set_idle_state()

    def _enter_stage(self) -> None:
        self._stage_start = time.time()
        self.config.stage_progress = 0.0
        stage = self._playlist[self._pos]
        self.config.stage = stage
        self._notify_state(0.0)

    def _advance_stage(self) -> None:
        next_pos = self._pos + 1
        if next_pos < len(self._playlist):
            self._pos = next_pos
            self._enter_stage()
        else:
            SequencerSettings.stop.fire(self.config)

    # -- Callbacks -----------------------------------------------------------

    def add_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        self._state_callbacks.add(callback)

    def remove_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        self._state_callbacks.discard(callback)

    def _notify_state(self, elapsed: float) -> None:
        state = SequencerState(
            stage=self.config.stage,
            stage_progress=self.config.stage_progress,
            progress=self.config.progress,
            elapsed=elapsed,
            active=self._active,
        )
        for cb in self._state_callbacks:
            try:
                cb(state)
            except Exception as e:
                logger.error(f"Sequencer state callback error: {e}")

    # -- Tick (called every render frame) ------------------------------------

    def update(self) -> None:
        """Advance sequencer state. Call once per frame."""
        if not self._active:
            return

        elapsed = time.time() - self._stage_start
        duration = self._stage_durations[self._pos]

        # Stage progress
        self.config.stage_progress = min(elapsed / duration, 1.0) if duration > 0 else 1.0

        # Overall progress
        if self._total_duration > 0:
            self.config.progress = min(
                (self._cumulative[self._pos] + elapsed) / self._total_duration, 1.0
            )
        else:
            self.config.progress = 1.0

        # State callbacks
        self._notify_state(elapsed)

        # Auto-advance
        if duration > 0 and elapsed >= duration:
            self._advance_stage()
