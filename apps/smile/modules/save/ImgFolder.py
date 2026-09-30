import cv2
import numpy as np
import threading
import queue
import os
import datetime
import shutil
import warnings

from modules.render.settings import SequencerSettings


class ImgFolder:
    """Saves the images of each visitor (capture, portrait, grin frames) as a folder for the players.
    A folder is built in temp/ and moved to images/ when its sequence ends complete."""

    def __init__(self, settings: SequencerSettings, max_queue_size=300, output_path: str = 'images/', temp_path: str = 'temp/', size: tuple[int, int] = (512, 640)) -> None:
        self.settings: SequencerSettings = settings
        self.save_path: str = output_path
        self.temp_path: str = temp_path
        self.size: tuple[int, int] = size   # saved size (4:5 as the portrait); the players show these rotated in 640x512 cells
        self.id: str = ''
        self.temp_dir: str = ''
        self.sequence_index: int = 0
        self.sequence_count: int = 0
        self.sequence_lock = threading.Lock()
        self.queue = queue.Queue(max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._save_loop)
        self.thread.start()
        self.settings.bind(SequencerSettings.save, self._on_save)

    def _save_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                image: np.ndarray = self.queue.get(timeout=.1)
                scaled_image: np.ndarray = self._scale_image(image)
                path: str = ''
                with self.sequence_lock:
                    path = os.path.join(self.temp_dir, str(self.sequence_index).zfill(3) + ".jpg")
                    self.sequence_index += 1
                if path:
                    cv2.imwrite(path, scaled_image)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error occurred while saving image: {e}")

    def save_image(self, image: np.ndarray) -> None:
        if not self.settings.save:
            return
        self.queue.put(image)

    def new_sequence(self) -> None:
        if not self.settings.save:
            return
        self._new_sequence()

    def end_sequence(self, keep: bool) -> None:
        """Close the visitor's folder: kept for the players when the sequence is complete, else deleted."""
        if not self.settings.save:
            return
        self._end_sequence(keep)

    def _on_save(self, value: bool) -> None:
        # switching off drops the unfinished folder, switching on starts a fresh one
        if value:
            self._new_sequence()
        else:
            self._end_sequence(keep=False)

    def _new_sequence(self) -> None:
        with self.sequence_lock:
            self.sequence_index = 0
            self._empty_queue()
            self.id: str = self._create_sequence_id(self.sequence_count)
            self.temp_dir = os.path.join(self.temp_path, self.id)
            if os.path.exists(self.temp_dir):
                warnings.warn(f"Directory {self.temp_dir} already exists. Erasing contents.")
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir)

    def _end_sequence(self, keep: bool) -> None:
        with self.sequence_lock:
            if self.sequence_index > 0:
                self._empty_queue()
                if keep:
                    destination_folder: str = os.path.join(self.save_path, self.id)
                    shutil.move(self.temp_dir, destination_folder)
                    self.sequence_count += 1
                else:
                    shutil.rmtree(self.temp_dir)

                self.sequence_index = 0

    def stop(self) -> None:
        self.settings.unbind(SequencerSettings.save, self._on_save)
        self.stop_event.set()
        self.thread.join()

    def _create_sequence_id(self, odd_or_even: int) -> str:
        timestamp: str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # YearMonthDayHourMinuteSecond
        if odd_or_even % 2 == 0:
            timestamp += "_A"
        else:
            timestamp += "_B"
        return timestamp

    def _empty_queue(self) -> None:
        with self.queue.mutex:  # Use queue's internal mutex for thread safety
            self.queue.queue.clear()

    def _scale_image(self, image: np.ndarray) -> np.ndarray:
        if (image.shape[1], image.shape[0]) == self.size:
            return image
        scaled_image: np.ndarray = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        return scaled_image