"""Prerecorded pose sequence player for INTRO stages.

Loads an HDF5 recording via :class:`Player` and provides pose frames
through a lightweight :class:`SequenceDataProxy` that can be passed to
:class:`CentreGeometry` in place of a full :class:`DataHub`.
"""
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Any

from modules.pose.frame import Frame, FrameDict
from modules.pose.recorder.player import Player
from modules.utils import Color

from .settings import IntroSequenceSettings

logger = logging.getLogger(__name__)


class SequenceDataProxy:
    """Minimal duck-typed stand-in for the blackboard.

    Only implements :meth:`get_frame`, which is the sole method
    :class:`CentreGeometry` calls on its ``board`` reference.
    """

    def __init__(self) -> None:
        self._frames: FrameDict = {}

    def update(self, frames: FrameDict) -> None:
        """Replace the stored frame dict."""
        self._frames = frames

    def clear(self) -> None:
        """Remove all stored frames."""
        self._frames = {}

    def get_frame(self, stage: int, track_id: int) -> Any | None:
        """Return the frame for *track_id*, ignoring *stage*."""
        return self._frames.get(track_id)


class FixedColorProxy:
    """Duck-typed stand-in for :class:`ColorSettings`.

    Returns the same color for every track, so the intro overlay
    renders with a single configurable color on all screens.
    """

    def __init__(self, settings: IntroSequenceSettings) -> None:
        self._settings = settings

    @property
    def track_colors(self) -> list[Color]:
        return [self._settings.color] * 16


class IntroSequencePlayer:
    """Plays a single recorded track across all cameras during INTRO.

    Wraps :class:`Player` to load HDF5 pose recordings and provides
    a per-frame ``update()`` that returns a :class:`FrameDict` with the
    selected ``source_track`` duplicated for each camera slot.

    The recording plays once from start to end.  After the last recorded
    frame the player deactivates and ``update()`` returns an empty dict.
    """

    def __init__(self, settings: IntroSequenceSettings, num_cams: int) -> None:
        self._settings = settings
        self._num_cams = num_cams
        self._player = Player()
        self._active = False
        self._start_time = 0.0
        self._duration = 0.0
        self._time_offset = 0.0

        folder = Path(settings.recording_path)
        if folder.exists():
            self._player.load(folder)
            ts = self._player._all_timestamps
            if len(ts) > 0:
                self._time_offset = float(ts[0])
                self._duration = float(ts[-1]) - self._time_offset
                logger.info("IntroSequencePlayer loaded %.1fs from %s", self._duration, folder)
            else:
                logger.warning("IntroSequencePlayer: no frames in %s", folder)
        else:
            logger.warning("IntroSequencePlayer: folder not found: %s", folder)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def duration(self) -> float:
        return self._duration

    def start(self) -> None:
        """Begin playback from the start of the recording."""
        if self._duration <= 0.0:
            return
        self._active = True
        self._start_time = time.time()
        if self._settings.verbose:
            logger.info("IntroSequencePlayer started")

    def stop(self) -> None:
        """Stop playback."""
        if self._active and self._settings.verbose:
            logger.info("IntroSequencePlayer stopped")
        self._active = False

    def update(self) -> FrameDict:
        """Return current frame dict, mapped to all camera slots.

        Returns an empty dict when inactive or past the recording end.
        """
        if not self._active:
            return {}

        elapsed = time.time() - self._start_time
        if elapsed >= self._duration:
            self._active = False
            if self._settings.verbose:
                logger.info("IntroSequencePlayer finished (%.1fs)", elapsed)
            return {}

        query_ts = self._time_offset + elapsed
        source = self._player.get_frame_dict(query_ts)
        source_track = self._settings.source_track

        frame = source.get(source_track)
        if frame is None:
            return {}

        # Duplicate the source track to each camera slot
        result: FrameDict = {}
        for cam_id in range(self._num_cams):
            result[cam_id] = Frame(
                cam_id, cam_id, frame.time_stamp, dict(frame._features)
            )
        return result
