import numpy as np
from json import loads
from time import sleep
from threading import Thread, Lock
from typing import Callable



class PersonMessage():
    def __init__(self, found: bool, numPersons: int, image : np.ndarray | None = None) -> None:
        self.found: bool = found
        self.numPersons: int = numPersons
        self.image: np.ndarray | None = image

MessageCallback = Callable[[PersonMessage], None]

class PersonFinder(Thread):
    def __init__(self) -> None:
        super().__init__()
        self._input_mutex: Lock = Lock()
        self._thread_running: bool = False
        self._image: np.ndarray | None = None
        self._message_callbacks: list[MessageCallback] = []

        self._confidence: float = 0.85
        self._size_filter: float = 0.1
        self._edge_filter: float = 0.15
        self._bottom_filter: float = 0.99

    def stop(self) -> None:
        if (self._thread_running):
            self._thread_running = False
            self.join()

    def run(self) -> None:
        self._thread_running = True

        try:
            from ultralytics import YOLO
        except ImportError:
            print('ultralytics not installed, person finder disabled')
            return

        model = YOLO('files/yolov8n.pt')
        while self._thread_running:
            image: np.ndarray | None = self.get_image()
            self.set_image(None)
            if image is not None:
                results = model.track(source=image, stream=False, show=False, conf=self._confidence, imgsz= 640, verbose = False, persist = True, classes=0)
                json_string: str | None = results[0].tojson(True)
                if json_string:
                    res_dict: list = loads(json_string)
                    num_persons: int = len(res_dict)
                    person_found: bool = self._filter(res_dict)
                    person_image: np.ndarray | None = None
                    if person_found:
                        person_image = image #image.copy()
                    self.message_callback(PersonMessage(person_found, num_persons, person_image))
            else:
                sleep(0.01)

    def _filter(self, dict: list) -> bool:
        for r in dict:
            left: float = r['box']['x1']
            right: float = r['box']['x2']
            width: float = right - left
            centre: float = (left + right) * 0.5
            bottom: float = r['box']['y2']
            if bottom > self._bottom_filter:
                if width > self._size_filter:
                    if centre > self._edge_filter:
                        if centre < 1.0 - self._edge_filter:
                            return True
        return False

    def set_image(self, image: np.ndarray | None) -> None:
        with self._input_mutex:
            self._image = image
    def get_image(self) -> np.ndarray | None:
        with self._input_mutex:
            return self._image

    def set_confidence(self, value: float) -> None:
        self._confidence = value
    def get_confidence(self) -> float:
        return self._confidence

    def set_size_filter(self, value: float) -> None:
        self._size_filter = value
    def get_size_filter(self) -> float:
        return self._size_filter

    def set_edge_filter(self, value: float) -> None:
        self._edge_filter = value
    def get_edge_filter(self) -> float:
        return self._edge_filter

    def set_bottom_filter(self, value: float) -> None:
        self._bottom_filter = value
    def get_bottom_filter(self) -> float:
        return self._bottom_filter

    def message_callback(self, message: PersonMessage) -> None:
        with self._input_mutex:
            for c in self._message_callbacks: c(message)

    def add_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            self._message_callbacks.append(callback)

    def remove_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            if callback in self._message_callbacks:
                self._message_callbacks.remove(callback)

    def clear_message_callbacks(self) -> None:
        with self._input_mutex:
            self._message_callbacks = []
