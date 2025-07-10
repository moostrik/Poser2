# Standard library imports
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Optional, Callable

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from modules.person.Person import Person, TrackingStatus
from modules.pose.PoseDefinitions import JointAngleDict, PoseAngleKeypoints, Keypoint
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

# Type for analysis output callback
@dataclass
class PoseWindowData:
    window_id: int
    angles: pd.DataFrame
    confidences: pd.DataFrame

PoseWindowCallback = Callable[[PoseWindowData], None]
PoseWindowDataDict = dict[int, PoseWindowData]

@dataclass
class PoseWindowVisualisationData:
    window_id: int
    mesh_data: np.ndarray  # shape: (num_frames, num_joints, 2)

PoseWindowVisualisationCallback = Callable[[PoseWindowVisualisationData], None]

class PoseWindowBuffer(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()

        # Input
        self.person_input_queue: Queue[Person] = Queue()

        # Windowing for joint angles
        self.window_size: int = int(settings.pose_window_size * settings.camera_fps)
        self.angle_windows: dict[int, pd.DataFrame] = {}
        self.conf_windows: dict[int, pd.DataFrame] = {}

        # Callbacks for analysis output
        self.callback_lock = Lock()
        self.output_callbacks: set[PoseWindowCallback] = set()

        hot_reloader = HotReloadMethods(self.__class__, True)

    def stop(self) -> None:
        self._stop_event.set()
        with self.callback_lock:
            self.output_callbacks.clear()
        self.angle_windows.clear()
        self.conf_windows.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                person: Optional[Person] = self.person_input_queue.get(block=True, timeout=0.01)
                if person is not None:
                    try:
                        self._process(person)
                    except Exception as e:
                        print(f"Error processing person {person.id}: {e}")
                    self.person_input_queue.task_done()
            except Empty:
                continue

    def add_person(self, person: Person) -> None:
        self.person_input_queue.put(person)

    def _process(self, person: Person) -> None:
        """ Process a person and update the joint angle windows. """

        if person.status == TrackingStatus.REMOVED or person.status == TrackingStatus.NEW:
            # print(f"Skipping person {person.id} with status {person.status}")
            # If the person is removed or lost, clear their angle and confidence windows
            self.angle_windows.pop(person.id, None)
            self.conf_windows.pop(person.id, None)
            return

        if person.pose_angles is None:
            print(f"WINDOW: Skipping person {person.id} with no pose angles, this should not happen")
            return

        # Build angle/confidence dicts
        angle_row: dict[str, float] = {Keypoint(k).name: v["angle"] for k, v in person.pose_angles.items()}
        conf_row: dict[str, float] = {Keypoint(k).name: v["confidence"] for k, v in person.pose_angles.items()}
        timestamp: pd.Timestamp = person.time_stamp  # Assume pd.Timestamp

        # Update angle window
        angle_df: pd.DataFrame = self.angle_windows.get(person.id, pd.DataFrame())
        angle_row_df = pd.DataFrame([angle_row], index=[timestamp])
        angle_df = pd.concat([angle_df, angle_row_df])
        angle_df.sort_index(inplace=True)
        angle_df = angle_df.iloc[-self.window_size:]
        self.angle_windows[person.id] = angle_df

        # # Update confidence window
        conf_df: pd.DataFrame = self.conf_windows.get(person.id, pd.DataFrame())
        conf_row_df = pd.DataFrame([conf_row], index=[timestamp])
        conf_df = pd.concat([conf_df, conf_row_df])
        conf_df.sort_index(inplace=True)
        conf_df = conf_df.iloc[-self.window_size:]
        self.conf_windows[person.id] = conf_df

        # Interpolate and smooth angles
        angle_df.interpolate(method='time', limit_direction='both', limit = 7, inplace=True)
        # angle_df = PoseWindowBuffer.rolling_circular_mean(angle_df, window=0.4, min_periods=1)
        angle_df = PoseWindowBuffer.ewm_circular_mean(angle_df, span=7.0)

        # Notify callbacks with both DataFrames
        self._notify_callbacks(PoseWindowData(person.id, angle_df, conf_df))

    @staticmethod
    def rolling_circular_mean(df: pd.DataFrame, window: float = 0.3, min_periods: int = 1) -> pd.DataFrame:
        """ Rolling mean on unwrapped angles to avoid discontinuities at ±π. """
        window_str: str = f"{int(window * 1000)}ms"
        # Unwrap angles to remove discontinuities
        df_unwrapped: pd.DataFrame = df.apply(np.unwrap)
        # Rolling mean on unwrapped angles
        df_smooth: pd.DataFrame = df_unwrapped.rolling(window=window_str, min_periods=min_periods).mean()

        # Wrap back to [-pi, pi]
        return ((df_smooth + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def ewm_circular_mean(df: pd.DataFrame, span: float = 5.0) -> pd.DataFrame:
        """Exponential moving average on unwrapped angles to avoid discontinuities at ±π."""
        # Unwrap angles to avoid discontinuities at ±π
        df_unwrapped: pd.DataFrame = df.apply(np.unwrap)

        # Apply exponential moving average on unwrapped data
        df_smooth: pd.DataFrame = df_unwrapped.ewm(span=span, adjust=False).mean()

        # Wrap back to [-π, π]
        return ((df_smooth + np.pi) % (2 * np.pi)) - np.pi

    def add_window_callback(self, callback: PoseWindowCallback) -> None:
        """ Register a callback to receive the current pandas DataFrame window. """
        with self.callback_lock:
            self.output_callbacks.add(callback)

    def _notify_callbacks(self, data: PoseWindowData) -> None:
        """ Handle the output of the analysis. """
        with self.callback_lock:
            for callback in self.output_callbacks:
                callback(data)
