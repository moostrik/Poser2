# Standard library imports
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Optional, Callable

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from modules.pose.PoseDefinitions import *
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

# Type for analysis output callback
@dataclass (frozen=False)
class PoseStreamData:
    player_id: int
    angles: pd.DataFrame
    confidences: pd.DataFrame

PoseStreamDataCallback = Callable[[PoseStreamData], None]
PoseStreamDataDict = dict[int, PoseStreamData]

class PoseStreamProcessor(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()

        # Input
        self.pose_input_queue: Queue[Pose] = Queue()

        # Windowing for joint angles
        self.buffer_capacity: int = int(settings.pose_buffer_duration * settings.camera_fps)
        self.angle_buffers: dict[int, pd.DataFrame] = {}
        self.confidence_buffers: dict[int, pd.DataFrame] = {}

        # Callbacks for analysis output
        self.callback_lock = Lock()
        self.output_callbacks: set[PoseStreamDataCallback] = set()

        hot_reloader = HotReloadMethods(self.__class__, True)

    def stop(self) -> None:
        self._stop_event.set()
        with self.callback_lock:
            self.output_callbacks.clear()
        self.angle_buffers.clear()
        self.confidence_buffers.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pose: Optional[Pose] = self.pose_input_queue.get(block=True, timeout=0.01)
                if pose is not None:
                    try:
                        self._process(pose)
                    except Exception as e:
                        print(f"Error processing pose {pose.id}: {e}")
                    self.pose_input_queue.task_done()
            except Empty:
                continue

    def add_pose(self, pose: Pose) -> None:
        self.pose_input_queue.put(pose)

    def _process(self, pose: Pose) -> None:
        """ Process a pose and update the joint angle windows. """

        if pose.angles is None:
            # print(f"WINDOW: Skipping pose {pose.id} with no pose angles, this should not happen")
            return

        # Build angle/confidence dicts
        angle_row: dict[str, float] = {Keypoint(k).name: v["angle"] for k, v in pose.angles.items()}
        conf_row: dict[str, float] = {Keypoint(k).name: v["confidence"] for k, v in pose.angles.items()}
        timestamp: pd.Timestamp = pose.time_stamp  # Assume pd.Timestamp

        # Update angle window
        angle_df: pd.DataFrame = self.angle_buffers.get(pose.id, pd.DataFrame())
        angle_row_df = pd.DataFrame([angle_row], index=[timestamp])
        angle_df = pd.concat([angle_df, angle_row_df])
        angle_df.sort_index(inplace=True)
        angle_df = angle_df.iloc[-self.buffer_capacity:]
        self.angle_buffers[pose.id] = angle_df

        # # Update confidence window
        conf_df: pd.DataFrame = self.confidence_buffers.get(pose.id, pd.DataFrame())
        conf_row_df = pd.DataFrame([conf_row], index=[timestamp])
        conf_df = pd.concat([conf_df, conf_row_df])
        conf_df.sort_index(inplace=True)
        conf_df = conf_df.iloc[-self.buffer_capacity:]
        self.confidence_buffers[pose.id] = conf_df

        # Interpolate and smooth angles
        angle_df.interpolate(method='time', limit_direction='both', limit = 7, inplace=True)
        # angle_df = PoseWindowBuffer.rolling_circular_mean(angle_df, window=0.4, min_periods=1)
        angle_df = PoseStreamProcessor.ewm_circular_mean(angle_df, span=7.0)

        # Notify callbacks with both DataFrames
        self._notify_callbacks(PoseStreamData(pose.id, angle_df, conf_df))

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

    def add_stream_callback(self, callback: PoseStreamDataCallback) -> None:
        """ Register a callback to receive the current pandas DataFrame window. """
        with self.callback_lock:
            self.output_callbacks.add(callback)

    def _notify_callbacks(self, data: PoseStreamData) -> None:
        """ Handle the output of the analysis. """
        with self.callback_lock:
            for callback in self.output_callbacks:
                callback(data)
