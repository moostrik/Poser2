# Standard library imports
import time
from dataclasses import dataclass
from itertools import combinations
from threading import Event, Thread, Lock

# Third-party imports
import pandas as pd

# Local application imports
from modules.pose.PoseWindowBuffer import PoseWindowData
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

PoseWindowDict = dict[int, PoseWindowData]

@dataclass
class AnglePair:
    id_1: int
    id_2: int
    angles_1: pd.DataFrame
    angles_2: pd.DataFrame
    confidences_1: pd.DataFrame
    confidences_2: pd.DataFrame

class PoseAnalysis(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()
        self.data_lock = Lock()
        self.pose_windows: PoseWindowDict = {}

        self.analysis_interval: float = 1.0 / settings.analysis_rate_hz
        self.max_window_size: int = int(settings.pose_window_size * settings.camera_fps)
        self.analysis_window_size: int = min(int(settings.analysis_window_size * settings.camera_fps), self.max_window_size)

        self.round: str = f'{int(1000 / settings.camera_fps)}ms'  # Round to nearest 45ms
        self.nan_ratio: float = 0.7  # Minimum valid ratio of non-NaN values in a window
        self.max_age: float = 1.0

        hot_reloader = HotReloadMethods(self.__class__)

    def set_window(self, data: PoseWindowData) -> None:
        with self.data_lock:
            self.pose_windows[data.window_id] = data

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.analysis_interval)
            with self.data_lock:
                pose_windows: PoseWindowDict = self.pose_windows.copy()
            try:
                pose_windows = self._prepare_windows(pose_windows)
                self._analyse(pose_windows)
            except Exception as e:
                print(f"Error during analysis: {e}")
                continue

    def _prepare_windows(self, windows: PoseWindowDict) -> PoseWindowDict:
        """Filter windows based on time and length."""
        windows = self._filter_windows_by_time(windows, self.max_age)
        windows = self._filter_windows_by_length(windows, self.analysis_window_size)
        windows = self._trim_windows_to_length(windows, self.analysis_window_size)
        windows = self._filter_windows_by_nan(windows, self.nan_ratio)
        # for data in windows.values():
        #     data.angles.index = pd.to_datetime(data.angles.index).round(self.round)
        #     data.confidences.index = pd.to_datetime(data.confidences.index).round(self.round)
        return windows

    def _analyse(self, windows: PoseWindowDict) -> None:

        angle_pairs: list[AnglePair] = self._generate_overlapping_angle_pairs(windows)

        for pair in angle_pairs:
            print(f"Aligned {pair.id_1} and {pair.id_2}: {len(pair.angles_1)} overlapping frames, of {len(windows[pair.id_1].angles)} and {len(windows[pair.id_2].angles)} total frames")



    @staticmethod
    def _filter_windows_by_time(windows: PoseWindowDict, max_age_s: float = 2.0) -> PoseWindowDict:
        """Return only windows whose last timestamp is within max_age_s seconds from now."""
        now: pd.Timestamp = pd.Timestamp.now()
        filtered: PoseWindowDict = {}
        for window_id, data in windows.items():
            if not data.angles.empty:
                last_time = data.angles.index[-1]
                # print(last_time, (now - last_time).total_seconds())
                if (now - last_time).total_seconds() <= max_age_s:
                    filtered[window_id] = data
        return filtered

    @staticmethod
    def _filter_windows_by_length(windows: PoseWindowDict, min_length: int = 20) -> PoseWindowDict:
        """Return only windows with at least min_length frames."""
        return {wid: data for wid, data in windows.items() if len(data.angles) >= min_length}

    @staticmethod
    def _filter_windows_by_nan(windows: PoseWindowDict, min_valid_ratio: float = 0.7) -> PoseWindowDict:
        """Return only windows where the ratio of non-NaN values is above min_valid_ratio."""
        filtered: PoseWindowDict = {}
        for wid, data in windows.items():
            total: int = data.angles.size
            valid: int = data.angles.count().sum()
            if total > 0 and (valid / total) >= min_valid_ratio:
                filtered[wid] = data
        return filtered

    @staticmethod
    def _trim_windows_to_length(windows: PoseWindowDict, max_length: int ) -> PoseWindowDict:
        """ Trim each window's DataFrames to the last max_length frames. """
        for data in windows.values():
            if len(data.angles) > max_length:
                data.angles = data.angles.iloc[-max_length:]
                data.confidences = data.confidences.iloc[-max_length:]
        return windows

    @staticmethod
    def _generate_overlapping_angle_pairs(windows: PoseWindowDict) -> list[AnglePair]:
        """Generate all unique pairs of windows with overlapping time ranges."""
        angle_pairs: list[AnglePair] = []
        window_items: list[tuple[int, PoseWindowData]] = list(windows.items())

        for (id1, win1), (id2, win2) in combinations(window_items, 2):
            # Find overlapping time range
            t1_start, t1_end = win1.angles.index[0], win1.angles.index[-1]
            t2_start, t2_end = win2.angles.index[0], win2.angles.index[-1]
            overlap_start = max(t1_start, t2_start)
            overlap_end = min(t1_end, t2_end)
            if overlap_start >= overlap_end:
                continue

            # Slice both DataFrames to the overlapping interval
            angles1_overlap = win1.angles.loc[(win1.angles.index >= overlap_start) & (win1.angles.index <= overlap_end)]
            angles2_overlap = win2.angles.loc[(win2.angles.index >= overlap_start) & (win2.angles.index <= overlap_end)]
            confidences1_overlap = win1.confidences.loc[(win1.confidences.index >= overlap_start) & (win1.confidences.index <= overlap_end)]
            confidences2_overlap = win2.confidences.loc[(win2.confidences.index >= overlap_start) & (win2.confidences.index <= overlap_end)]

            angle_pairs.append(
                AnglePair(
                    id_1=id1,
                    id_2=id2,
                    angles_1=angles1_overlap,
                    angles_2=angles2_overlap,
                    confidences_1=confidences1_overlap,
                    confidences_2=confidences2_overlap
                )
            )
        return angle_pairs
