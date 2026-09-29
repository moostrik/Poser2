import cv2
import numpy as np
import threading
import queue
import os
import datetime
import shutil
import warnings


class ImgFolder:
    def __init__(self, max_queue_size=100, output_path: str = 'images/', temp_path: str = 'temp/', min_files: int = 10) -> None:
        self.active: bool = True
        self.save_path: str = output_path
        self.temp_path: str = temp_path
        self.scale: float = 0.5
        self.id: str = ''
        self.temp_dir: str = ''
        self.min_files: int = min_files
        self.sequence_index: int = 0
        self.sequence_count: int = 0
        self.sequence_lock = threading.Lock()
        self.queue = queue.Queue(max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._save_loop)
        self.thread.start()

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
        if not self.active:
            return
        self.queue.put(image)

    def new_sequence(self) -> None:
        if not self.active:
            return
        with self.sequence_lock:
            self.sequence_index = 0
            self._empty_queue()
            self.id: str = self._create_sequence_id(self.sequence_count)
            self.temp_dir = os.path.join(self.temp_path, self.id)
            if os.path.exists(self.temp_dir):
                warnings.warn(f"Directory {self.temp_dir} already exists. Erasing contents.")
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir)

    def end_sequence(self) -> None:
        if not self.active:
            return
        with self.sequence_lock:
            if self.sequence_index > 0:
                self._empty_queue()
                if self.sequence_index >= self.min_files:
                    destination_folder: str = os.path.join(self.save_path, self.id)
                    shutil.move(self.temp_dir, destination_folder)
                    self.sequence_count += 1
                else:
                    # print("Not enough images to save sequence" , self.sequence_index, " instead of ", self.min_files)
                    shutil.rmtree(self.temp_dir)

                self.sequence_index = 0

    def set_active(self, value: bool) -> None:
        if self.active and not value:
            self.end_sequence()
        if not self.active and value:
            self.new_sequence()
        self.active = value

    def stop(self) -> None:
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
        scaled_image: np.ndarray = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return scaled_image