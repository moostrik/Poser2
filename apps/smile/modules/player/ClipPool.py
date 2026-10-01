import logging
import os
import queue
import random
import threading
from dataclasses import dataclass

import numpy as np

from modules.video import ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class Clip:
    id: str
    frames: np.ndarray      # (frames, height, width, 3) bgr uint8
    fps: float
    style: str = ''         # the folder the clip was saved in: A or B
    plays: int = 0          # times started on a cell (a new clip starts level with the least played)
    active: int = 0         # cells playing it now
    fresh: bool = True      # not shown yet: the next cell that finishes takes it


def next_start(clips: list[Clip], started: set[str]) -> Clip | None:
    """The first clip in load order that has not started a cell in this round; None when all have."""
    for clip in clips:
        if clip.id not in started:
            return clip
    return None


def choose(clips: list[Clip], exclude: Clip | None = None, recent: int = 0) -> Clip | None:
    """The clip a cell plays next: one that is on the fewest cells now, of those a clip that was
    never shown, else one that was played the least, of those at random. `exclude` (the clip the
    cell just played) is left out unless it is the only one. So every clip is on the wall about
    equally, none repeats in its cell and a new visitor is shown at once.

    Before all that, the `recent` newest clips (the id is the recording time) are always on the
    wall: one of them that is on no cell is taken first. The cell that just played it leaves it to
    the next cell that finishes, so it moves around the wall."""
    candidates: list[Clip] = [clip for clip in clips if clip is not exclude] or clips
    if not candidates:
        return None
    newest: list[Clip] = sorted(clips, key=lambda clip: clip.id, reverse=True)[:max(recent, 0)]
    missing: list[Clip] = [clip for clip in newest if clip.active == 0 and clip is not exclude]
    if missing:
        candidates = missing
    active: int = min(clip.active for clip in candidates)
    candidates = [clip for clip in candidates if clip.active == active]
    candidates = [clip for clip in candidates if clip.fresh] or candidates
    plays: int = min(clip.plays for clip in candidates)
    return random.choice([clip for clip in candidates if clip.plays == plays])


def evict_id(clips: dict[str, Clip], style: str | None) -> str:
    """The clip to drop when the pool is over its size: the oldest of another style than the
    current one, else the oldest. A new visitor's clip (the newest id) is never the one."""
    others: list[str] = [id for id, clip in clips.items() if style is not None and clip.style != style]
    return min(others) if others else min(clips)


class ClipPool:
    """Clips decoded once into RAM at the cell size; playback never decodes.

    The scanner decides which files to load and adds them; one decode thread works through them,
    the last added first, so a new visitor never waits behind a reload. The pool only limits the
    number. With a `style` set (the players follow SMILE's style) the cells get clips of that style
    and the other style's clips are the first to go."""

    def __init__(self, size: int, cell_size: tuple[int, int], hwdecode: int | None = None, recent: int = 0) -> None:
        self.size: int = size
        self.recent: int = recent           # the newest clips that are always on the wall
        self.cell_size: tuple[int, int] = cell_size
        self.hwdecode: int | None = hwdecode
        self.style: str | None = None       # None: any style

        self._clips: dict[str, Clip] = {}
        self._lock = threading.Lock()
        self._pending: queue.LifoQueue[tuple[str, str] | None] = queue.LifoQueue()
        self._outstanding: int = 0      # files added and not yet decoded or skipped
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._decode_loop, name='ClipPool', daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._pending.put(None)
        self._thread.join()

    # SCANNER THREAD

    def add(self, path: str, style: str = '') -> None:
        with self._lock:
            self._outstanding += 1
        self._pending.put((path, style))

    def remove(self, id: str) -> None:
        with self._lock:
            self._clips.pop(id, None)

    def set_style(self, style: str | None) -> None:
        with self._lock:
            self.style = style

    # PLAYER AND LOADER THREADS

    def clips(self) -> list[Clip]:
        """The clips in the order they were loaded; those of the current style when there are any."""
        with self._lock:
            return self._candidates()

    def _candidates(self) -> list[Clip]:
        clips: list[Clip] = list(self._clips.values())
        if self.style is not None:
            current: list[Clip] = [clip for clip in clips if clip.style == self.style]
            if current:
                return current      # none loaded yet (just after a switch): any clip, so no cell goes empty
        return clips

    @property
    def idle(self) -> bool:
        """Nothing is waiting to be decoded."""
        with self._lock:
            return self._outstanding == 0

    def use(self, clip: Clip) -> None:
        """A cell starts on this clip (the player's hand-out at startup)."""
        with self._lock:
            self._take(clip)

    def release(self, clip: Clip) -> None:
        """A cell stopped playing this clip without taking a next one."""
        with self._lock:
            clip.active -= 1

    def get_clip(self, previous: Clip | None = None) -> Clip | None:
        """The next clip for a cell that just played `previous` (see `choose`); None while the pool
        is empty. Taking the next and giving up the previous is one step, so two cells that finish
        together never both take the same free clip."""
        with self._lock:
            clip: Clip | None = choose(self._candidates(), previous, self.recent)
            if previous is not None:
                previous.active -= 1
            if clip is not None:
                self._take(clip)
            return clip

    @staticmethod
    def _take(clip: Clip) -> None:
        clip.plays += 1
        clip.active += 1
        clip.fresh = False

    def spread(self) -> tuple[int, int]:
        """How the clips are spread over the cells now: (clips on at least one cell, most cells on one clip)."""
        with self._lock:
            actives: list[int] = [clip.active for clip in self._clips.values()]
        return sum(1 for a in actives if a > 0), max(actives, default=0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._clips)

    # DECODE THREAD

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._pending.get()
            if item is None or self._stop_event.is_set():
                continue
            path, style = item
            id: str = os.path.splitext(os.path.basename(path))[0]
            if self._is_wanted(id):
                clip: Clip | None = self._decode(id, path, style)
                if clip is not None:
                    self._insert(clip)
            with self._lock:
                self._outstanding -= 1

    def _is_wanted(self, id: str) -> bool:
        with self._lock:
            return id not in self._clips

    def _decode(self, id: str, path: str, style: str) -> Clip | None:
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
        return Clip(id, frames, fps, style)

    def _insert(self, clip: Clip) -> None:
        with self._lock:
            # level with the least played, so it is not shown more than the others once it has been shown
            clip.plays = min((c.plays for c in self._clips.values() if not c.fresh), default=0)
            clip.fresh = True
            self._clips[clip.id] = clip
            while len(self._clips) > self.size:
                del self._clips[evict_id(self._clips, self.style)]     # a loader mid-clip keeps its own reference
        logger.info('Clip %s %s: %d frames at %g fps, %d in the pool', clip.style, clip.id, len(clip.frames), clip.fps, len(self))
