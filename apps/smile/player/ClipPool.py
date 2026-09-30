import logging
import os
import queue
import random
import threading
from dataclasses import dataclass, field

import numpy as np

from modules.video import ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class Clip:
    id: str
    frames: np.ndarray      # (frames, height, width, 3) bgr uint8
    fps: float
    usage_count: int = field(default=0)


class ClipPool:
    """The newest clips, decoded once into RAM at the cell size; playback never decodes.

    The scanner adds and removes files; one decode thread works through them newest first. `ready`
    is set when the initial batch is decoded, so the cells do not all start on the first clip in."""

    def __init__(self, size: int, cell_size: tuple[int, int], hwdecode: int | None = None, max_usage_count: int = 10) -> None:
        self.size: int = size
        self.cell_size: tuple[int, int] = cell_size
        self.hwdecode: int | None = hwdecode
        self.max_usage_count: int = max_usage_count
        self.ready = threading.Event()

        self._clips: dict[str, Clip] = {}
        self._lock = threading.Lock()
        self._pending: queue.LifoQueue[str] = queue.LifoQueue()
        self._initial_done: bool = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._decode_loop, name='ClipPool', daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._pending.put('')
        self._thread.join()

    # SCANNER THREAD

    def add(self, path: str) -> None:
        self._pending.put(path)

    def remove(self, id: str) -> None:
        with self._lock:
            self._clips.pop(id, None)

    def initial_done(self) -> None:
        """The initial listing is queued; ready follows when it is decoded (at once when it was empty)."""
        self._initial_done = True
        if self._pending.empty():
            self.ready.set()

    # LOADER THREADS

    def get_clip(self) -> Clip | None:
        """A random clip, favouring the least played ones; None while the pool is empty."""
        with self._lock:
            if not self._clips:
                return None
            ids: list[str] = list(self._clips.keys())
            weights: list[float] = [1.0 / (self._clips[id].usage_count + 1) for id in ids]
            clip: Clip = self._clips[random.choices(ids, weights=weights, k=1)[0]]
            clip.usage_count = min(clip.usage_count + 1, self.max_usage_count)
            return clip

    def __len__(self) -> int:
        with self._lock:
            return len(self._clips)

    # DECODE THREAD

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            path: str = self._pending.get()
            if not path or self._stop_event.is_set():
                continue
            id: str = os.path.splitext(os.path.basename(path))[0]
            if self._is_wanted(id):
                clip: Clip | None = self._decode(id, path)
                if clip is not None:
                    self._insert(clip)
            if self._initial_done and self._pending.empty():
                self.ready.set()

    def _is_wanted(self, id: str) -> bool:
        with self._lock:
            if id in self._clips:
                return False
            return len(self._clips) < self.size or id > min(self._clips)

    def _decode(self, id: str, path: str) -> Clip | None:
        try:
            frames, fps = ffmpeg.decode_clip(path, self.cell_size, self.hwdecode)
        except Exception as e:
            if self.hwdecode is None:
                logger.warning('Skipping %s: %s', id, e)
                return None
            logger.warning('Hardware decode failed on %s, decoding in software: %s', id, e)
            try:
                frames, fps = ffmpeg.decode_clip(path, self.cell_size, None)
            except Exception as e2:
                logger.warning('Skipping %s: %s', id, e2)
                return None
        return Clip(id, frames, fps)

    def _insert(self, clip: Clip) -> None:
        with self._lock:
            self._clips[clip.id] = clip
            while len(self._clips) > self.size:
                del self._clips[min(self._clips)]     # a loader mid-clip keeps its own reference
        logger.info('Clip %s: %d frames at %g fps, %d in the pool', clip.id, len(clip.frames), clip.fps, len(self))
