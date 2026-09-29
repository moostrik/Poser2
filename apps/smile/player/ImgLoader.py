import os
import random
from threading import Thread, Lock, Event
import time
import cv2
import numpy as np
from typing import Callable
from player.FolderScanner import FolderScanner, imagePathInfo

ImageCallback= Callable[[str, np.ndarray, bool], None]

LoaderCallback= Callable[[int], None]


class LoadImages(Thread):
    def __init__(self, id: str, folder_scanner: FolderScanner, interval: float = 1.0, end_wait: float = 3.0) -> None:
        super().__init__()
        self._callback_mutex: Lock = Lock()
        self._stop_event: Event = Event()
        self._message_callbacks: list[ImageCallback] = []
        self._loader_callbacks: list[LoaderCallback] = []

        self._id: str = id
        self.scanner: FolderScanner = folder_scanner
        self._image_interval: float = interval
        self._last_image_wait_low: float = end_wait
        self._last_image_wait_high: float = end_wait * 1.2
        self._end_time: float = 0.0

        self._image_cntr: int = 0
        self._image_path_info: imagePathInfo = self.scanner.get_new_images()


    def run(self) -> None:

        self._end_time = time.time() + random.uniform(0.0, self._last_image_wait_high)
        while not self._stop_event.is_set():
            image_files: list[str] = self._image_path_info.images
            if self._image_cntr >= len(image_files):
                self._image_path_info: imagePathInfo = self.scanner.get_new_images()
                image_files = self._image_path_info.images
                self._image_cntr = 0

                panning: int = random.randint(0, 2)
                self._loader_callback(panning)

            # if time.time() >= self._end_time:
            image: np.ndarray | None = None
            try:
                image = cv2.imread(image_files[self._image_cntr])
            except Exception as e:
                print(f'Error loading image {image_files[self._image_cntr]}: {e}')

            if image is not None:
                self._image_callback(image, self._image_cntr == 0)
                pass

            if self._image_cntr >= len(image_files) - 1:
                wait_time = float(random.uniform(self._last_image_wait_low, self._last_image_wait_high))
                self._end_time += wait_time
            else:
                self._end_time += self._image_interval

            self._image_cntr += 1

            while time.time() < self._end_time and not self._stop_event.is_set():
                time.sleep(0.01)

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
