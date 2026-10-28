import random
from threading import Thread, Lock, Event
import time
import numpy as np
from typing import Any, Callable
from apps.smile.player.ClipPool import ClipPool, Clip

ImageCallback= Callable[[str, np.ndarray, bool], None]

LoaderCallback= Callable[[int], None]     # the cell a play starts in

def play_rate(fps: float, clip_fps: float, spread: float) -> float:
    """The speed of one play: fps (the clip's own rate when 0), off by a random fraction up to spread.
    Drawn per play, so the cells drift against each other and no cell is always the fast one."""
    rate: float = fps if fps > 0 else clip_fps
    spread = min(max(spread, 0.0), 0.9)
    return rate * random.uniform(1.0 - spread, 1.0 + spread)


class LoadImages(Thread):
    """One cell: stays empty until the player assigns its first clip, then plays clips from the pool
    on its own clock, one after the other, at the speed the settings say at each play (`fps`, `spread`).
    `sector` is the cell's third of the sphere, which the pool keeps every recent recording on."""

    def __init__(self, id: str, pool: ClipPool, settings: Any, sector: int = 0) -> None:
        super().__init__()
        self._callback_mutex: Lock = Lock()
        self._stop_event: Event = Event()
        self._message_callbacks: list[ImageCallback] = []
        self._loader_callbacks: list[LoaderCallback] = []

        self._id: str = id
        self.pool: ClipPool = pool
        self.settings: Any = settings   # fps (0 plays each clip at its own rate) and spread (a fraction)
        self.sector: int = sector
        self._end_time: float = 0.0
        self._first: Clip | None = None
        self._assigned: Event = Event()

    def assign(self, clip: Clip) -> None:
        """Start this cell on its first clip (any thread)."""
        self._first = clip
        self._assigned.set()

    @property
    def assigned(self) -> bool:
        return self._assigned.is_set()

    def run(self) -> None:
        while not self._assigned.wait(0.05):
            if self._stop_event.is_set():
                return
        clip: Clip | None = self._first
        self._first = None
        self._end_time = time.time()
        while not self._stop_event.is_set():
            if clip is None:
                self._sleep_until(time.time() + 1.0)
                self._end_time = time.time()
                clip = self.pool.get_clip(sector=self.sector)
                continue
            self._play(clip)
            if self._stop_event.is_set():
                self.pool.release(clip, self.sector)
                return
            # the next clip is one the other cells show the least, never the one just played; the
            # reference to that one is dropped here (the pool may have evicted it meanwhile)
            clip = self.pool.get_clip(clip, self.sector)

    def _play(self, clip: Clip) -> None:
        interval: float = 1.0 / play_rate(float(self.settings.fps), clip.fps, float(self.settings.spread))
        count: int = len(clip.frames)
        # the cell this play starts in: the Max patch places the laugh by the cell's screen
        self._loader_callback(int(self._id))
        for i in range(count):
            if self._stop_event.is_set():
                return
            self._image_callback(clip.frames[i], i == 0)
            self._end_time += interval
            now: float = time.time()
            if now > self._end_time + interval:     # fell behind: resync instead of bursting
                self._end_time = now
            self._sleep_until(self._end_time)

    def _sleep_until(self, t: float) -> None:
        while time.time() < t and not self._stop_event.is_set():
            time.sleep(0.005)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()

    # CALLBACKS
    def _image_callback(self, image: np.ndarray, initial: bool = False) -> None:
        with self._callback_mutex:
            for c in self._message_callbacks:
                c(self._id, image, initial)

    def add_image_callback(self, callback: ImageCallback) -> None:
        with self._callback_mutex:
            self._message_callbacks.append(callback)

    def remove_image_callback(self, callback: ImageCallback) -> None:
        with self._callback_mutex:
            if callback in self._message_callbacks:
                self._message_callbacks.remove(callback)

    def clear_image_callbacks(self) -> None:
        with self._callback_mutex:
            self._message_callbacks = []

    def get_image_callbacks(self) -> list[ImageCallback]:
        with self._callback_mutex:
            return self._message_callbacks.copy()


    def _loader_callback(self, cell: int) -> None:
        with self._callback_mutex:
            for c in self._loader_callbacks:
                c(cell)

    def add_loader_callback(self, callback: LoaderCallback) -> None:
        with self._callback_mutex:
            self._loader_callbacks.append(callback)

    def remove_loader_callback(self, callback: LoaderCallback) -> None:
        with self._callback_mutex:
            if callback in self._loader_callbacks:
                self._loader_callbacks.remove(callback)

    def clear_loader_callbacks(self) -> None:
        with self._callback_mutex:
            self._loader_callbacks = []

    def get_loader_callbacks(self) -> list[LoaderCallback]:
        with self._callback_mutex:
            return self._loader_callbacks.copy()
