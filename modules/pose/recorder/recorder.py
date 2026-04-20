"""Recorder — records FrameDict updates to chunked HDF5 files.

Chunking is driven externally by firing the shared ``split`` button
(e.g. from a Session timer or manually).  The recorder reacts to:
  - ``start`` (button) — start recording
  - ``stop`` (button) — stop recording
  - ``split`` (button) — flush current chunk and start a new one
  - ``name`` — appended to the timestamped folder name
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


def make_folder_name(name: str = "") -> str:
    folder = time.strftime("%Y%m%d-%H%M%S")
    if name:
        folder += f"_{name}"
    return folder


def make_file_name(prefix: str, chunk: int, suffix: str) -> str:
    return f"{prefix}_{chunk:03d}{suffix}"


# Sentinel objects placed in the queue by split / stop
class _Split:
    pass

class _Stop:
    pass

_SPLIT = _Split()
_STOP  = _Stop()

_MAX_BUFFER_FRAMES = 50_000


class Recorder:
    """Records FrameDict updates to chunked HDF5 files.

    Controlled entirely through settings fields:
      - ``start`` (button) — start recording
      - ``stop`` (button) — stop recording
      - ``split`` (button) — chunk boundary
      - ``name`` — appended to folder timestamp
    """

    def __init__(self, settings: RecorderSettings) -> None:
        self.settings = settings
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._folder: Path | None = None
        self._recording_start: float = 0.0

        settings.bind(RecorderSettings.start, self._on_start)
        settings.bind(RecorderSettings.stop, self._on_stop)
        settings.bind(RecorderSettings.split, self._on_split)
        settings.bind(RecorderSettings.enabled, self._on_enabled)

    # ── Settings callbacks ───────────────────────────────────────────────

    def _on_enabled(self, value: bool) -> None:
        if not value:
            self._on_stop()

    def _on_start(self, _=None) -> None:
        if not self.settings.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        folder_name = make_folder_name(self.settings.name)
        folder = Path(self.settings.output_path) / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        self._start(folder, time.time())
        self.settings.recording = True

    def _on_stop(self, _=None) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop()
        self.settings.recording = False

    def _on_split(self, _=None) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(_SPLIT)

    # ── Internal lifecycle ───────────────────────────────────────────────

    def _start(self, folder: Path, recording_start_time: float) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("PoseRecorder._start() called while already recording — ignored")
            return
        self._folder = folder
        self._recording_start = recording_start_time
        self._thread = threading.Thread(target=self._run, daemon=True, name="PoseRecorder")
        self._thread.start()
        logger.info("PoseRecorder started → %s", folder)

    def _stop(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put(_STOP)
        self._thread.join()
        self._thread = None
        logger.info("PoseRecorder stopped")

    def submit_frames(self, stage: int, frame_dict: 'FrameDict') -> None:
        """Callback — enqueue a shallow copy of the FrameDict if recording and stage matches."""
        if stage != self.settings.stage:
            return
        if self._thread is not None and self._thread.is_alive():
            self._queue.put((time.time(), dict(frame_dict)))

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
                if len(buffer) >= _MAX_BUFFER_FRAMES:
                    logger.warning("PoseRecorder buffer full (%d frames) — dropping incoming frames", _MAX_BUFFER_FRAMES)
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
        path = folder / make_file_name("pose", chunk_index, ".h5")
        try:
            write_chunk(path, buffer, self._recording_start, chunk_start, chunk_index, feature_types)
            logger.debug("PoseRecorder wrote %s (%d frames)", path.name, len(buffer))
        except Exception:
            logger.exception("PoseRecorder failed to write %s", path)
