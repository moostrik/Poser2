# Standard library imports
from dataclasses import dataclass
from multiprocessing import Process, Queue, Event, Lock, Manager
from threading import Thread
from typing import Optional, Callable, Set
import signal
import sys

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from modules.pose.PoseDefinitions import *
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods




@dataclass (frozen=True)
class PoseStreamInput:
    id: int
    time_stamp: pd.Timestamp
    angles: Optional[JointAngleDict]
    is_final: bool

    @classmethod
    def from_pose(cls, pose: Pose) -> 'PoseStreamInput':
        return cls(
            id=pose.id,
            time_stamp=pose.time_stamp,
            angles=pose.angles,
            is_final=pose.is_final
        )

# Type for analysis output callback
@dataclass (frozen=False)
class PoseStreamData:
    id: int
    angles: pd.DataFrame
    confidences: pd.DataFrame

PoseStreamDataCallback = Callable[[PoseStreamData], None]
PoseStreamDataDict = dict[int, PoseStreamData]

class PoseStreamManager:
    def __init__(self, settings) -> None:
        self.settings = settings  # Store settings for processor recreation
        self.result_queue = Queue()
        self.processor = PoseStreamProcessor(settings, self.result_queue)
        self.output_callbacks: Set[PoseStreamDataCallback] = set()
        self.result_thread = Thread(target=self._handle_results, daemon=True)
        self.running = False

        # Hot reload setup for restarting processor
        self.hot_reloader = HotReloadMethods(PoseStreamProcessor, True, True)
        self.hot_reloader.add_file_changed_callback(self._on_file_changed)

    def _on_file_changed(self) -> None:
        """Restart the processor when files change."""
        print("[PoseStream] File changed, restarting processor...")
        if self.running:
            self._restart_processor()

    def _restart_processor(self) -> None:
        """Restart the processor process."""
        try:
            # Stop current processor
            if self.processor.is_alive():
                print("[PoseStream] Stopping current processor...")
                self.processor.stop()
                self.processor.join(timeout=2.0)  # Wait a bit longer for graceful shutdown

                if self.processor.is_alive():
                    print("[PoseStream] Force terminating processor...")
                    self.processor.terminate()
                    self.processor.join(timeout=1.0)

            # Create new processor
            print("[PoseStream] Creating new processor...")
            self.processor = PoseStreamProcessor(self.settings, self.result_queue)
            self.processor.start()
            print("[PoseStream] Processor restarted successfully")

        except Exception as e:
            print(f"[PoseStream] Error restarting processor: {e}")

    def start(self) -> None:
        """Start the processor and result handler."""
        self.running = True
        self.processor.start()
        self.result_thread.start()

    def stop(self) -> None:
        """Stop the processor and result handler."""
        self.running = False

        # Stop hot reloader
        self.hot_reloader.stop_file_watcher()

        # Stop processor
        self.processor.stop()
        self.processor.join(timeout=1.0)
        if self.processor.is_alive():
            self.processor.terminate()

    def add_pose(self, pose) -> None:
        """Add pose to processing queue."""
        try:
            pose_stream_input: PoseStreamInput = PoseStreamInput.from_pose(pose)
            self.processor.add_pose(pose_stream_input)
        except Exception as e:
            print(f"[PoseStream] Error adding pose: {e}")
            # # If processor is dead, try to restart it
            # if not self.processor.is_alive() and self.running:
            #     self._restart_processor()

    def add_stream_callback(self, callback: PoseStreamDataCallback) -> None:
        """Register a callback to receive processed data."""
        self.output_callbacks.add(callback)

    def _handle_results(self) -> None:
        """Handle results from the processor in the main process."""
        while self.running:
            try:
                data: PoseStreamData = self.result_queue.get(timeout=0.1)
                # Call all callbacks with the data
                for callback in self.output_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"Error in callback: {e}")
            except:
                continue


class PoseStreamProcessor(Process):
    def __init__(self, settings: Settings, result_queue: Queue) -> None:
        super().__init__()

        # Use multiprocessing primitives
        self._stop_event = Event()
        self.pose_input_queue: Queue[PoseStreamInput] = Queue()

        # For sending results back to main process
        self.result_queue = result_queue if result_queue else Queue()

        # Store settings values (not the settings object itself)
        self.buffer_capacity: int = int(settings.pose_buffer_duration * settings.camera_fps)
        self.resample_interval: str = f"{int(1.0 / settings.camera_fps * 1000)}ms"

        # Initialize buffers (will be recreated in child process)
        self.angle_buffers: dict[int, pd.DataFrame] = {}
        self.confidence_buffers: dict[int, pd.DataFrame] = {}

    def stop(self) -> None:
        self._stop_event.set()
        self.angle_buffers.clear()
        self.confidence_buffers.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pose: Optional[PoseStreamInput] = self.pose_input_queue.get(block=True, timeout=0.01)
                if pose is not None:
                    try:
                        self._process(pose)
                    except Exception as e:
                        print(f"Error processing pose {pose.id}: {e}")
            except:
                continue

    def add_pose(self, pose: PoseStreamInput) -> None:
        """Add pose to processing queue - can be called from main process."""
        try:
            self.pose_input_queue.put(pose, block=False)
        except:
            # Queue is full, skip this pose
            pass

    def _notify_callbacks(self, data: PoseStreamData) -> None:
        """ Send results back to main process via queue. """
        try:
            self.result_queue.put(data, block=False)
        except:
            pass

    def _process(self, pose: PoseStreamInput) -> None:
        """ Process a pose and update the joint angle windows. """

        if pose.angles is None:
            print(f"[PoseStreamProcessor] Pose {pose.id} has no angles, skipping")
            return

        if pose.is_final:
            if self.angle_buffers.get(pose.id) is not None:
                self.angle_buffers.pop(pose.id, None)
                self.confidence_buffers.pop(pose.id, None)
            return

        time_stamp: pd.Timestamp = pose.time_stamp

        # Update angle window
        angle_df: pd.DataFrame = self.angle_buffers.get(pose.id, pd.DataFrame(columns=PoseAngleNames))
        angle_row: pd.Series = pd.Series(
            {Keypoint(k).name: v["angle"] for k, v in pose.angles.items()},
            index=PoseAngleNames  # Ensures all columns are present, missing ones will be NaN
        )
        angle_df.loc[time_stamp] = angle_row # type: ignore
        angle_df.sort_index(inplace=True)
        angle_df = angle_df.iloc[-self.buffer_capacity:]
        self.angle_buffers[pose.id] = angle_df

        # Update confidence window
        conf_df: pd.DataFrame = self.confidence_buffers.get(pose.id, pd.DataFrame(columns=PoseAngleNames))
        conf_row: pd.Series = pd.Series(
            {Keypoint(k).name: v["confidence"] for k, v in pose.angles.items()},
            index=PoseAngleNames  # Ensures all columns are present, missing ones will be NaN
        )
        conf_df.loc[time_stamp] = conf_row  # type: ignore
        conf_df.sort_index(inplace=True)
        conf_df = conf_df.iloc[-self.buffer_capacity:]
        self.confidence_buffers[pose.id] = conf_df

        interpolated: pd.DataFrame = angle_df.interpolate(method='time', limit_direction='both', limit=7)
        smoothed: pd.DataFrame = PoseStreamProcessor.ewm_circular_mean(interpolated, span=7.0)
        self._notify_callbacks(PoseStreamData(pose.id, smoothed, conf_df))

    @staticmethod
    def rolling_circular_mean(df: pd.DataFrame, window: float = 0.3, min_periods: int = 1) -> pd.DataFrame:
        """ Rolling mean on unwrapped angles to avoid discontinuities at ±π. """
        window_str: str = f"{int(window * 1000)}ms"
        # Unwrap angles to remove discontinuities
        df_unwrapped: pd.DataFrame = df.apply(PoseStreamProcessor.safe_unwrap)
        # Rolling mean on unwrapped angles
        df_smooth: pd.DataFrame = df_unwrapped.rolling(window=window_str, min_periods=min_periods).mean()

        # Wrap back to [-pi, pi]
        return ((df_smooth + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def ewm_circular_mean(df: pd.DataFrame, span: float = 5.0) -> pd.DataFrame:
        """Exponential moving average on unwrapped angles to avoid discontinuities at ±π."""
        df_unwrapped: pd.DataFrame = df.apply(PoseStreamProcessor.safe_unwrap)

        # Apply exponential moving average on unwrapped data
        df_smooth: pd.DataFrame = df_unwrapped.ewm(span=span, adjust=False).mean()

        # Wrap back to [-π, π]
        return ((df_smooth + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def get_mean_interval(df: pd.DataFrame) -> float:
        """Calculate the mean interval between timestamps in a DataFrame."""
        if df.empty:
            return 0.0
        intervals: pd.Series = df.index.to_series().diff().dt.total_seconds()
        return intervals.mean() if not intervals.empty else 0.0

    @staticmethod
    def safe_unwrap(series):
        # Find first and last valid indices
        valid_indices = series.dropna().index
        if len(valid_indices) < 2:
            return series  # Not enough valid values to unwrap

        # Create result series
        result = series.copy()

        # Only unwrap the valid range
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        valid_slice = series.loc[first_valid:last_valid]

        # Unwrap only the valid slice
        unwrapped_slice = pd.Series(np.unwrap(valid_slice.values), index=valid_slice.index)

        # Put back in result
        result.loc[first_valid:last_valid] = unwrapped_slice

        return result