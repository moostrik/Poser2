"""Recorder — records FrameDict updates to chunked HDF5 files.

Two operating modes:

  Linked (synced to video):
    main.py wires the video Recorder's lifecycle callbacks to
    ``recorder.start / split / stop``.  Chunks align with video chunks.

  Standalone:
    The user presses the Start button in the settings UI.  The recorder
    creates its own timestamped folder and writes a single ``pose_000.h5``
    (no splits since there is no external split signal).
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .frame_io import write_chunk
from .settings import RecorderSettings

if TYPE_CHECKING:
    from modules.pose.frame.frame import FrameDict
    from modules.pose.features.base import BaseFeature

import logging
logger = logging.getLogger(__name__)


# Sentinel objects placed in the queue by split() / stop()
class _Split:
    pass

class _Stop:
    pass

_SPLIT = _Split()
_STOP  = _Stop()


class Recorder:
    """Records FrameDict updates to chunked HDF5 files.

    Usage (linked to video recorder, wired in main.py):
        video_recorder.add_recording_start_callback(recorder.start)
        video_recorder.add_recording_split_callback(recorder.split)
        video_recorder.add_recording_stop_callback(recorder.stop)
        point_extractor.add_frames_callback(recorder.on_frame_dict)

    Usage (standalone — driven by settings UI buttons):
        point_extractor.add_frames_callback(recorder.on_frame_dict)
        # Start / Stop via recording.start / Stop buttons in settings panel.
    """

    def __init__(self, settings: RecorderSettings) -> None:
        self.settings = settings
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._folder: Path | None = None
        self._recording_start: float = 0.0

        settings.bind(RecorderSettings.start, self._on_settings_start)
        settings.bind(RecorderSettings.stop,  self._on_settings_stop)

    # ── Public API (called by video recorder callbacks or settings UI) ──

    def start(self, folder: Path, recording_start_time: float) -> None:
        """Begin recording to *folder*.  Safe to call while idle only."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("PoseRecorder.start() called while already recording — ignored")
            return
        self._folder = folder
        self._recording_start = recording_start_time
        self._thread = threading.Thread(target=self._run, daemon=True, name="PoseRecorder")
        self._thread.start()
        self.settings.recording = True
        logger.info("PoseRecorder started → %s", folder)

    def stop(self) -> None:
        """Stop recording; blocks until the background thread has flushed."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put(_STOP)
        self._thread.join()
        self._thread = None
        self.settings.recording = False
        logger.info("PoseRecorder stopped")

    def split(self) -> None:
        """Signal a chunk boundary.  The background thread flushes the current
        buffer and starts a new chunk.  Non-blocking."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(_SPLIT)

    def on_frame_dict(self, frame_dict: 'FrameDict') -> None:
        """Callback — enqueue a shallow copy of the FrameDict if recording."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put((time.time(), dict(frame_dict)))

    # ── Settings UI handlers (standalone mode) ──────────────────────────

    def _on_settings_start(self, _=None) -> None:
        folder = Path(self.settings.recordings_path) / f"pose_{time.strftime('%Y%m%d-%H%M%S')}"
        folder.mkdir(parents=True, exist_ok=True)
        self.start(folder, time.time())

    def _on_settings_stop(self, _=None) -> None:
        self.stop()

    # ── Background thread ────────────────────────────────────────────────

    def _active_feature_types(self) -> list[type['BaseFeature']]:
        from modules.pose.features import FEATURE_CLASS
        return [FEATURE_CLASS[f] for f in self.settings.features]

    def _run(self) -> None:
        buffer: list[tuple[float, 'FrameDict']] = []
        chunk_index = 0
        chunk_start = self._recording_start
        folder = self._folder

        while True:
            item = self._queue.get()

            if isinstance(item, _Stop):
                if buffer and folder is not None:
                    self._flush(folder, buffer, chunk_start, chunk_index)
                break

            elif isinstance(item, _Split):
                if buffer and folder is not None:
                    self._flush(folder, buffer, chunk_start, chunk_index)
                chunk_index += 1
                chunk_start = time.time()
                buffer = []

            else:
                buffer.append(item)

    def _flush(
        self,
        folder: Path,
        buffer: list[tuple[float, 'FrameDict']],
        chunk_start: float,
        chunk_index: int,
    ) -> None:
        feature_types = self._active_feature_types()
        if not feature_types:
            return
        path = folder / f"pose_{chunk_index:03d}.h5"
        try:
            write_chunk(path, buffer, self._recording_start, chunk_start, chunk_index, feature_types)
            logger.debug("PoseRecorder wrote %s (%d frames)", path.name, len(buffer))
        except Exception:
            logger.exception("PoseRecorder failed to write %s", path)
