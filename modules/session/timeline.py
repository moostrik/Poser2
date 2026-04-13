"""Timeline — stage-based sequencer with config-driven control, ticked by the render loop."""

import time
from typing import Callable, Set

from modules.settings import BaseSettings, Field, Widget

import logging
logger = logging.getLogger(__name__)


class TimelineSettings(BaseSettings):
    """Base configuration for Timeline.

    Subclass this per project and override:
    - ``stages`` with your stage enum list (the playlist of stages to play)
    - ``durations`` with your per-stage durations
    - ``stage`` with your stage enum type (read-only output)

    Example::

        class ShowTimelineSettings(TimelineSettings):
            stages:    Field[list[ShowStage]] = Field(list(ShowStage), widget=Widget.playlist)
            durations: Field[list[float]]     = Field([3.0, 10.0, 5.0])
            stage:     Field[ShowStage]       = Field(ShowStage.START, access=Field.READ)
    """
    stages:         Field[list[int]]   = Field([0], description="Stages to play")
    durations:      Field[list[float]] = Field([0.0], min=0.0, description="Duration per stage (seconds)")
    run:            Field[bool]        = Field(False, newline=True)
    loop:           Field[bool]        = Field(False, widget=Widget.switch, description="Loop timeline when all stages complete")
    skip:           Field[bool]        = Field(False, widget=Widget.button, description="Skip to next stage")
    stage:          Field[int]         = Field(0, access=Field.READ, description="Current stage")
    stage_progress: Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Stage progress")
    progress:       Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Overall progress")


class Timeline:
    """Tick-based timeline that progresses through a playlist of stages.

    The ``stages`` list defines which stages to play and in what order.
    Each entry indexes into ``durations`` for that stage's length.
    Call ``update()`` every frame (e.g. from ``data_hub.notify_update()``).

    Example::

        config = ShowTimelineSettings()
        timeline = Timeline(config)
        timeline.add_stage_callback(lambda s: print(f"Stage: {s}"))
        data_hub.add_update_callback(timeline.update)

        config.stages = [0, 1, 2, 3, 4]   # Full show
        config.run = True

        config.stages = [2]                # Test single stage
        config.run = True
    """

    def __init__(self, config: TimelineSettings) -> None:
        self.config = config

        if not config.durations:
            raise ValueError("config.durations must contain at least one entry")

        # Callbacks
        self._stage_callbacks: Set[Callable] = set()
        self._time_callbacks: Set[Callable[[float], None]] = set()

        # Runtime state
        self._active = False
        self._playlist_pos: int = 0
        self._stage_start: float = 0.0
        self._updating_run = False
        self._updating_stage = False

        self._setup_watchers()

    def _setup_watchers(self) -> None:
        self.config.bind(TimelineSettings.run, self._on_run_change)
        self.config.bind(TimelineSettings.skip, self._on_skip)

    def _on_run_change(self, value: bool) -> None:
        if self._updating_run:
            return
        if value and not self._active:
            self._start_show()
        elif not value and self._active:
            self._stop_show()

    def _on_skip(self, value: bool) -> None:
        if value and self._active:
            self._advance_stage()

    # -- Playlist helpers ----------------------------------------------------

    @property
    def _playlist(self) -> list[int]:
        return list(self.config.stages)

    @property
    def _current_stage_index(self) -> int:
        """The actual stage index at the current playlist position."""
        playlist = self._playlist
        if not playlist or self._playlist_pos >= len(playlist):
            return 0
        return int(playlist[self._playlist_pos])

    def _get_stage_duration(self, stage_index: int) -> float:
        durations = self.config.durations
        if 0 <= stage_index < len(durations):
            return float(durations[stage_index])
        return 0.0

    def _get_playlist_total_duration(self) -> float:
        return sum(self._get_stage_duration(int(s)) for s in self._playlist)

    def _get_playlist_elapsed_before(self) -> float:
        return sum(
            self._get_stage_duration(int(self._playlist[i]))
            for i in range(self._playlist_pos)
        )

    # -- Show lifecycle ------------------------------------------------------

    def _start_show(self) -> None:
        playlist = self._playlist
        if not playlist:
            logger.warning("Timeline: empty stages playlist, not starting")
            self._sync_run(False)
            return
        self._active = True
        self._playlist_pos = 0
        self._enter_stage()

    def _stop_show(self) -> None:
        self._active = False
        self.config.stage_progress = 0.0
        self.config.progress = 0.0
        self._sync_run(False)

    def _enter_stage(self) -> None:
        self._stage_start = time.time()
        self.config.stage_progress = 0.0
        stage_index = self._current_stage_index
        self._sync_stage(stage_index)
        self._notify_stage_callbacks(stage_index)

    def _advance_stage(self) -> None:
        next_pos = self._playlist_pos + 1
        playlist = self._playlist
        if next_pos < len(playlist):
            self._playlist_pos = next_pos
            self._enter_stage()
        elif self.config.loop:
            self._playlist_pos = 0
            self._enter_stage()
        else:
            self._stop_show()

    # -- Callbacks -----------------------------------------------------------

    def add_stage_callback(self, callback: Callable) -> None:
        self._stage_callbacks.add(callback)

    def remove_stage_callback(self, callback: Callable) -> None:
        self._stage_callbacks.discard(callback)

    def _notify_stage_callbacks(self, stage: int) -> None:
        for cb in self._stage_callbacks:
            try:
                cb(stage)
            except Exception as e:
                logger.error(f"Timeline: Error in stage callback: {e}")

    def add_time_callback(self, callback: Callable[[float], None]) -> None:
        self._time_callbacks.add(callback)

    def remove_time_callback(self, callback: Callable[[float], None]) -> None:
        self._time_callbacks.discard(callback)

    def _notify_time_callbacks(self, elapsed: float) -> None:
        for cb in self._time_callbacks:
            try:
                cb(elapsed)
            except Exception as e:
                logger.error(f"Timeline: Error in time callback: {e}")

    # -- Sync helpers --------------------------------------------------------

    def _sync_run(self, value: bool) -> None:
        if self.config.run != value:
            self._updating_run = True
            self.config.run = value
            self._updating_run = False

    def _sync_stage(self, index: int) -> None:
        if int(self.config.stage) != index:
            self._updating_stage = True
            self.config.stage = index
            self._updating_stage = False

    # -- Tick (called every render frame) ------------------------------------

    def update(self) -> None:
        """Advance timeline state. Call once per frame."""
        if not self._active:
            return

        now = time.time()
        stage_elapsed = now - self._stage_start
        stage_index = self._current_stage_index
        stage_duration = self._get_stage_duration(stage_index)

        # Stage progress (0-1)
        if stage_duration > 0:
            self.config.stage_progress = min(stage_elapsed / stage_duration, 1.0)
        else:
            self.config.stage_progress = 1.0

        # Overall progress across playlist (0-1)
        total = self._get_playlist_total_duration()
        if total > 0:
            self.config.progress = min(
                (self._get_playlist_elapsed_before() + stage_elapsed) / total, 1.0
            )
        else:
            self.config.progress = 1.0

        # Time callbacks
        self._notify_time_callbacks(stage_elapsed)

        # Auto-advance when duration reached
        if stage_duration > 0 and stage_elapsed >= stage_duration:
            self._advance_stage()
