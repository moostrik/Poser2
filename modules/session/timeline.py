"""Timeline — stage-based sequencer with config-driven control, ticked by the render loop."""

import time
from typing import Callable, Set

from modules.settings import BaseSettings, Field, Widget

import logging
logger = logging.getLogger(__name__)


class TimelineSettings(BaseSettings):
    """Base configuration for Timeline.

    Subclass this per project and add:
    - A ``durations`` override with your stage durations
    - Optionally a ``stage`` Field with your stage enum (access=READ)

    Example::

        class ShowTimelineSettings(TimelineSettings):
            durations: Field[list[float]] = Field([3.0, 10.0])
            stage:     Field[ShowStage]   = Field(ShowStage.START, access=Field.READ)
    """
    run:            Field[bool]        = Field(False, newline=True)
    loop:           Field[bool]        = Field(False, widget=Widget.switch, description="Loop timeline when all stages complete")
    skip:           Field[bool]        = Field(False, widget=Widget.button, description="Skip to next stage")
    durations:      Field[list[float]] = Field([0.0], min=0.0, description="Duration per stage (seconds)")
    stage_progress: Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Stage progress")
    progress:       Field[float]       = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Overall progress")


class Timeline:
    """Tick-based timeline that progresses through project-defined stages.

    Stage count and durations are read from ``config.durations``.
    Call ``update()`` every frame (e.g. from ``data_hub.notify_update()``).

    Example::

        config = ShowTimelineSettings()
        timeline = Timeline(config)
        timeline.add_stage_callback(lambda s: print(f"Stage: {s}"))
        data_hub.add_update_callback(timeline.update)
        config.run = True      # Begin show
        config.run = False     # Stop show
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
        self._stage_index: int = 0
        self._stage_start: float = 0.0
        self._updating_run = False

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

    # -- Show lifecycle ------------------------------------------------------

    def _start_show(self) -> None:
        self._active = True
        self._stage_index = 0
        self._enter_stage()

    def _stop_show(self) -> None:
        self._active = False
        self.config.stage_progress = 0.0
        self.config.progress = 0.0
        self._sync_run(False)

    def _enter_stage(self) -> None:
        self._stage_start = time.time()
        self.config.stage_progress = 0.0
        if hasattr(self.config, 'stage'):
            setattr(self.config, 'stage', self._stage_index)
        self._notify_stage_callbacks(self._stage_index)

    def _advance_stage(self) -> None:
        next_index = self._stage_index + 1
        if next_index < len(self.config.durations):
            self._stage_index = next_index
            self._enter_stage()
        elif self.config.loop:
            self._stage_index = 0
            self._enter_stage()
        else:
            self._stop_show()

    def _get_stage_duration(self, index: int) -> float:
        return float(self.config.durations[index])

    def _get_total_duration(self) -> float:
        return sum(self.config.durations)

    def _get_elapsed_before_current(self) -> float:
        return sum(self.config.durations[i] for i in range(self._stage_index))

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

    # -- Tick (called every render frame) ------------------------------------

    def update(self) -> None:
        """Advance timeline state. Call once per frame."""
        if not self._active:
            return

        now = time.time()
        stage_elapsed = now - self._stage_start
        stage_duration = self._get_stage_duration(self._stage_index)

        # Stage progress (0-1)
        if stage_duration > 0:
            self.config.stage_progress = min(stage_elapsed / stage_duration, 1.0)
        else:
            self.config.stage_progress = 1.0

        # Overall progress (0-1)
        total = self._get_total_duration()
        if total > 0:
            self.config.progress = min((self._get_elapsed_before_current() + stage_elapsed) / total, 1.0)
        else:
            self.config.progress = 1.0

        # Time callbacks
        self._notify_time_callbacks(stage_elapsed)

        # Auto-advance when duration reached
        if stage_duration > 0 and stage_elapsed >= stage_duration:
            self._advance_stage()
