import logging
import os
import queue
import random
import threading
from dataclasses import dataclass, field

import numpy as np

from .. import sphere
from modules import video

logger = logging.getLogger(__name__)


@dataclass(eq=False)        # a clip is itself, not its values: `in` and `==` never compare the frames
class Clip:
    id: str
    frames: np.ndarray      # (frames, height, width, 3) bgr uint8
    fps: float
    style: str = ''         # the folder the clip was saved in: A or B
    plays: int = 0          # times started on a cell (levelled when the clip joins the archive)
    active: int = 0         # cells playing it now
    on: list[int] = field(default_factory=lambda: [0] * sphere.SECTORS)     # of them, per third of the sphere


def next_start(clips: list[Clip], started: set[str]) -> Clip | None:
    """The first clip in load order that has not started a cell in this round; None when all have."""
    for clip in clips:
        if clip.id not in started:
            return clip
    return None


def choose(clips: list[Clip], exclude: Clip | None = None, recent: list[Clip] | tuple[Clip, ...] = (), sector: int = 0) -> Clip | None:
    """The clip a cell in `sector` (its third of the sphere) plays next. `exclude` (the clip the cell
    just played) is left out unless it is the only one, so no clip repeats in its cell.

    First, the `recent` clips (the last recordings) are always on every third of the sphere, so a
    visitor sees their video wherever they stand: one of them that is on no cell of this third is
    taken. The cell that just played it leaves it to the next cell that finishes there, so it moves
    around. Otherwise: a clip that is on the fewest cells now, of those one that was played the
    least, of those at random. So every clip is on the sphere about equally."""
    candidates: list[Clip] = [clip for clip in clips if clip is not exclude] or clips
    if not candidates:
        return None
    missing: list[Clip] = [clip for clip in recent if clip.on[sector] == 0 and clip is not exclude]
    if missing:
        candidates = missing
    active: int = min(clip.active for clip in candidates)
    candidates = [clip for clip in candidates if clip.active == active]
    plays: int = min(clip.plays for clip in candidates)
    return random.choice([clip for clip in candidates if clip.plays == plays])


class ClipPool:
    """Clips decoded once into RAM at the cell size; playback never decodes. Two lists:

    recent    the last `recent` recordings of the style, oldest to newest; always on the wall
    archive   the other clips, in the order they joined, oldest first

    A recording joins `recent`; when that holds too many, its oldest moves to `archive` as the
    newest there. A clip from disk (at a style switch, or recycled) joins `archive` as its newest.
    When the pool is over its size the oldest of `archive` leaves; a cell still playing it keeps
    its own reference. The scanner decides which files to load and adds them; one decode thread
    works through them, the last added first, so a new recording never waits behind a reload."""

    def __init__(self, size: int, cell_size: tuple[int, int], hwaccel: int | None = None, recent: int = 0) -> None:
        self.size: int = size
        self.recent: int = recent           # how many of the last recordings are kept in `recent`
        self.cell_size: tuple[int, int] = cell_size
        self.hwaccel: int | None = hwaccel
        self.style: str | None = None       # None: any style

        self._recent: list[Clip] = []
        self._archive: list[Clip] = []
        self._lock = threading.Lock()
        self._pending: queue.LifoQueue[tuple[str, str, bool] | None] = queue.LifoQueue()
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

    def add(self, path: str, style: str = '', recent: bool = False) -> None:
        """Load this file: a recording (`recent`) or a clip from disk for the archive."""
        with self._lock:
            self._outstanding += 1
        self._pending.put((path, style, recent))

    def remove(self, id: str) -> None:
        with self._lock:
            self._recent = [clip for clip in self._recent if clip.id != id]
            self._archive = [clip for clip in self._archive if clip.id != id]

    def switch(self, style: str | None) -> None:
        """The wall changes to this style. What is in the pool becomes the oldest of the archive
        (the last recordings at its end, so they leave last); the new style's clips join behind."""
        with self._lock:
            self._archive = self._archive + self._recent
            self._recent = []
            self.style = style

    def ids(self) -> set[str]:
        with self._lock:
            return {clip.id for clip in self._recent + self._archive}

    # PLAYER AND LOADER THREADS

    def clips(self) -> list[Clip]:
        """The clips of the current style, `recent` then `archive`; any clip while that style has none."""
        with self._lock:
            return self._candidates()

    def _candidates(self) -> list[Clip]:
        clips: list[Clip] = self._recent + self._archive
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

    def use(self, clip: Clip, sector: int = 0) -> None:
        """A cell in `sector` starts on this clip (the player's hand-out at startup)."""
        with self._lock:
            self._take(clip, sector)

    def release(self, clip: Clip, sector: int = 0) -> None:
        """A cell in `sector` stopped playing this clip without taking a next one."""
        with self._lock:
            self._drop(clip, sector)

    def get_clip(self, previous: Clip | None = None, sector: int = 0) -> Clip | None:
        """The next clip for a cell in `sector` that just played `previous` (see `choose`); None while
        the pool is empty. Taking the next and giving up the previous is one step, so two cells that
        finish together never both take the same free clip."""
        with self._lock:
            candidates: list[Clip] = self._candidates()
            recent: list[Clip] = [clip for clip in self._recent if clip in candidates]
            clip: Clip | None = choose(candidates, previous, recent, sector)
            if previous is not None:
                self._drop(previous, sector)
            if clip is not None:
                self._take(clip, sector)
            return clip

    @staticmethod
    def _take(clip: Clip, sector: int) -> None:
        clip.plays += 1
        clip.active += 1
        clip.on[sector] += 1

    @staticmethod
    def _drop(clip: Clip, sector: int) -> None:
        clip.active -= 1
        clip.on[sector] -= 1

    def spread(self) -> tuple[int, int]:
        """How the clips are spread over the cells now: (clips on at least one cell, most cells on one clip)."""
        with self._lock:
            actives: list[int] = [clip.active for clip in self._recent + self._archive]
        return sum(1 for a in actives if a > 0), max(actives, default=0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._recent) + len(self._archive)

    # DECODE THREAD

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._pending.get()
            if item is None or self._stop_event.is_set():
                continue
            self._load(*item)
            with self._lock:
                self._outstanding -= 1

    def _load(self, path: str, style: str, recent: bool) -> None:
        id: str = os.path.splitext(os.path.basename(path))[0]
        if self._is_stale(style):
            logger.debug('Not loading %s: the wall left style %s', id, style)
            return
        clip: Clip | None = self._find(id)
        if clip is not None:
            self._insert(clip, recent, reused=True)     # still in memory: no second decode
            return
        clip = self._decode(id, path, style)
        if clip is not None:
            self._insert(clip, recent)

    def _is_stale(self, style: str) -> bool:
        """A file queued for a style the wall has switched away from since."""
        with self._lock:
            return self.style is not None and style != self.style

    def _find(self, id: str) -> Clip | None:
        with self._lock:
            for clip in self._recent + self._archive:
                if clip.id == id:
                    return clip
        return None

    def _decode(self, id: str, path: str, style: str) -> Clip | None:
        try:
            frames, fps = video.decode_clip(path, self.cell_size, self.hwaccel)
        except Exception as e:
            if self.hwaccel is None:
                logger.warning('Skipping %s: %s', id, e)
                return None
            logger.warning('Hardware decode failed on %s, decoding in software: %s', id, e)
            try:
                frames, fps = video.decode_clip(path, self.cell_size, None)
            except Exception as e2:
                logger.warning('Skipping %s: %s', id, e2)
                return None
        return Clip(id, frames, fps, style)

    def _insert(self, clip: Clip, recent: bool = False, reused: bool = False) -> None:
        """The clip joins `recent` (a recording) or `archive` (from disk). One that is in the pool
        already moves to the list it is handed over for."""
        with self._lock:
            if self.style is not None and clip.style != self.style:
                return      # the wall switched while this one was being decoded
            self._recent = [c for c in self._recent if c is not clip]
            self._archive = [c for c in self._archive if c is not clip]
            if recent and self.recent > 0:
                self._recent.append(clip)
                self._recent.sort(key=lambda c: c.id)       # the id is the recording time
                while len(self._recent) > self.recent:
                    self._join_archive(self._recent.pop(0))
            else:
                self._join_archive(clip)
            while len(self._recent) + len(self._archive) > self.size:
                (self._archive or self._recent).pop(0)      # the oldest leaves; a loader mid-clip keeps its own reference
            count: int = len(self._recent) + len(self._archive)
            joined: str = 'recent' if clip in self._recent else 'archive'
        logger.info('Clip %s %s: %d frames at %g fps, %s%s, %d in the pool', clip.style, clip.id,
                    len(clip.frames), clip.fps, joined, ' (reused)' if reused else '', count)

    def _join_archive(self, clip: Clip) -> None:
        """Onto the end of the archive, level with its least played clip. A clip from `recent` was
        always on a cell and has many more plays; without the levelling it would lose every draw
        and sit off the wall for its whole turn in the archive."""
        clip.plays = min((c.plays for c in self._archive if c.style == clip.style), default=0)
        self._archive.append(clip)
