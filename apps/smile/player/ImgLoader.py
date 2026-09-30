import random
from threading import Thread, Lock, Event
import time
import numpy as np
from typing import Callable
from player.ClipPool import ClipPool, Clip

ImageCallback= Callable[[str, np.ndarray, bool], None]

LoaderCallback= Callable[[int], None]

READY_WAIT: float = 20.0    # longest wait for the pool's initial decode before a cell starts anyway


class LoadImages(Thread):
    """One cell: plays clips from the pool on its own clock, holding the last frame before the next clip."""

    def __init__(self, id: str, pool: ClipPool, end_wait: float = 3.0) -> None:
        super().__init__()
        self._callback_mutex: Lock = Lock()
        self._stop_event: Event = Event()
        self._message_callbacks: list[ImageCallback] = []
        self._loader_callbacks: list[LoaderCallback] = []

        self._id: str = id
        self.pool: ClipPool = pool
        self._last_image_wait_low: float = end_wait
        self._last_image_wait_high: float = end_wait * 1.2
        self._end_time: float = 0.0

    def run(self) -> None:
        self.pool.ready.wait(READY_WAIT)
        self._end_time = time.time() + random.uniform(0.0, self._last_image_wait_high)
        while not self._stop_event.is_set():
            clip: Clip | None = self.pool.get_clip()
            if clip is None:
                self._sleep_until(time.time() + 1.0)
                self._end_time = time.time()
                continue
            self._play(clip)
            clip = None     # release the frames: the pool may have dropped this clip meanwhile

    def _play(self, clip: Clip) -> None:
        interval: float = 1.0 / clip.fps
        count: int = len(clip.frames)
        self._loader_callback(random.randint(0, 2))
        for i in range(count):
            if self._stop_event.is_set():
                return
            self._image_callback(clip.frames[i], i == 0)
            if i == count - 1:
                self._end_time += random.uniform(self._last_image_wait_low, self._last_image_wait_high)
            else:
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


    def _loader_callback(self, panning: int) -> None:
        with self._callback_mutex:
            for c in self._loader_callbacks:
                c(panning)

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
