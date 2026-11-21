"""The walls' cells playing clips, all from one thread: a heap of the cells by their next frame's deadline, and one
`Waker` that waits for the earliest. At each wake the due cells hand their frame on and take their next deadline. Each
cell keeps its own clock: its speed per play (`fps`, `spread`), its clip order and its bang when a play starts."""

import heapq
import logging
import random
import time
from threading import Lock, Thread
from typing import Callable

import numpy as np

from modules.utils import Broadcast, Waker

from .clip_pool import Clip, ClipPool
from .settings import WallsSettings

logger = logging.getLogger(__name__)

RETRY_AFTER: float = 1.0        # s a cell without a clip waits before it asks the pool again

ImageCallback = Callable[[int, np.ndarray, bool], None]     # the cell, its frame, whether the play's first
PlayCallback = Callable[[int], None]                        # the cell a play starts in


def play_rate(fps: float, clip_fps: float, spread: float) -> float:
    """The speed of one play: fps (the clip's own rate when 0), off by a random fraction up to spread.
    Drawn per play, so the cells drift against each other and no cell is always the fast one."""
    rate: float = fps if fps > 0 else clip_fps
    spread = min(max(spread, 0.0), 0.9)
    return rate * random.uniform(1.0 - spread, 1.0 + spread)


class ClipCell:
    """One cell's play: its clip, the frame it shows next, the interval of this play, and its clock (`end_time`, the
    moment its next frame is due). Stepped by `ClipCells` only."""

    def __init__(self, id: int, sector: int) -> None:
        self.id: int = id
        self.sector: int = sector               # the cell's third of the sphere, which the pool keeps recent clips on
        self.clip: Clip | None = None
        self.index: int = 0                     # the frame shown next; past the last: the play is over
        self.interval: float = 0.0
        self.end_time: float = 0.0


class ClipCells(Thread):
    """See the module docstring. A cell stays empty until it is assigned its first clip (`assign`)."""

    def __init__(self, ids_sectors: list[tuple[int, int]], pool: ClipPool, settings: WallsSettings) -> None:
        super().__init__(name='ClipCells', daemon=True)
        self.pool: ClipPool = pool
        self.settings: WallsSettings = settings     # fps (0 plays each clip at its own rate) and spread (a fraction)
        self.cells: dict[int, ClipCell] = {id: ClipCell(id, sector) for id, sector in ids_sectors}
        self._heap: list[tuple[float, int]] = []    # (deadline, cell id): the playing cells by their next moment
        self._lock: Lock = Lock()                   # the heap: `assign` pushes from the player's start thread
        self._waker: Waker = Waker()
        self._running: bool = False
        self._image_callbacks: Broadcast[[int, np.ndarray, bool]] = Broadcast(name='cell image')
        self._play_callbacks: Broadcast[[int]] = Broadcast(name='cell play')

    def add_image_callback(self, callback: ImageCallback) -> None:
        self._image_callbacks.add_callback(callback)

    def add_play_callback(self, callback: PlayCallback) -> None:
        self._play_callbacks.add_callback(callback)

    def assign(self, id: int, clip: Clip) -> None:
        """Start a cell on its first clip (any thread): its first frame is due now."""
        cell: ClipCell = self.cells[id]
        now: float = time.perf_counter()
        cell.clip, cell.index, cell.end_time = clip, 0, now
        with self._lock:
            heapq.heappush(self._heap, (now, id))
        self._waker.set()

    def start(self) -> None:
        self._running = True
        super().start()

    def stop(self) -> None:
        self._running = False
        self._waker.set()
        if self.ident is not None:
            self.join()
        self._waker.close()
        for cell in self.cells.values():
            if cell.clip is not None:
                self.pool.release(cell.clip, cell.sector)
                cell.clip = None

    def run(self) -> None:
        while self._running:
            next_due: float | None = self.step(time.perf_counter())
            self._waker.wait(next_due if next_due is not None else time.perf_counter() + RETRY_AFTER)

    def step(self, now: float) -> float | None:
        """Every cell due by `now` shows its frame and takes its next deadline; the earliest deadline after, None
        while no cell plays."""
        while True:
            with self._lock:
                if not self._heap or self._heap[0][0] > now:
                    return self._heap[0][0] if self._heap else None
                _, id = heapq.heappop(self._heap)
            try:
                deadline: float = self._advance(self.cells[id], now)
            except Exception:
                logger.exception('Cell %d failed a frame', id)
                deadline = now + RETRY_AFTER
            with self._lock:
                heapq.heappush(self._heap, (deadline, id))

    def _advance(self, cell: ClipCell, now: float) -> float:
        """One due moment of a cell: the next clip when the play is over, then its frame; its next deadline."""
        if cell.clip is None or cell.index >= len(cell.clip.frames):
            # the next clip is one the other cells show the least, never the one just played
            cell.clip = self.pool.get_clip(cell.clip, cell.sector)
            cell.index = 0
            if cell.clip is None:
                cell.end_time = now + RETRY_AFTER
                return cell.end_time
        clip: Clip = cell.clip
        if cell.index == 0:
            cell.interval = 1.0 / play_rate(float(self.settings.fps), clip.fps, float(self.settings.spread))
            # the cell this play starts in: the Max patch places the laugh by the cell's screen
            self._play_callbacks(cell.id)
        self._image_callbacks(cell.id, clip.frames[cell.index], cell.index == 0)
        cell.index += 1
        cell.end_time += cell.interval
        if now > cell.end_time + cell.interval:     # fell behind: resync instead of bursting
            cell.end_time = now
        return cell.end_time
