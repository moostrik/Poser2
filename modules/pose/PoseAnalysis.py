# Standard library imports
from threading import Event, Thread, Lock
import time

# Third-party imports
import pandas as pd

# Local application imports
from modules.pose.PoseWindowBuffer import PoseWindowData
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

class PoseAnalysis(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()
        self.analysis_interval: float = 1.0 / settings.analysis_rate_hz
        self.max_window_size: int = int(settings.pose_window_size * settings.camera_fps)
        self.analysis_window_size: int = min(int(settings.analysis_window_size * settings.camera_fps), self.max_window_size)
        self.pose_windows: dict[int, PoseWindowData] = {}
        self.data_lock = Lock()

        hot_reloader = HotReloadMethods(self.__class__)

    def stop(self) -> None:
        self._stop_event.set()

    def set_window(self, data: PoseWindowData) -> None:
        with self.data_lock:
            self.pose_windows[data.window_id] = data
    def get_windows(self) -> dict[int, PoseWindowData]:
        with self.data_lock:
            return self.pose_windows.copy()

    def run(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.analysis_interval)
            data_items: dict[int, PoseWindowData] = self.get_windows()
            self.analyse(data_items)

    def analyse(self, data: dict[int, PoseWindowData]) -> None:
        """Perform correlation or other analysis on PoseWindowData."""
        # --- Your analysis logic here ---
        # print(f"Analysing window {data.window_id} with {len(data.angles)} frames")
        # Example: correlation = data.angles.corr()
        # ---------------------------------