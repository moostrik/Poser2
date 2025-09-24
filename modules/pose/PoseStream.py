# Standard library imports
import traceback
import signal
from dataclasses import dataclass
from multiprocessing import Process, Queue, Event
from queue import Empty
from threading import Thread
from time import sleep, perf_counter, time
from typing import Optional, Callable, Set

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
    angles: PoseAngleData | None
    is_removed: bool

    @classmethod
    def from_pose(cls, pose: Pose) -> 'PoseStreamInput':
        return cls(
            id=pose.id,
            time_stamp=pose.tracklet.time_stamp,
            angles=pose.angle_data,
            is_removed=pose.tracklet.is_removed
        )

# Type for analysis output callback
@dataclass (frozen=True)
class PoseStreamData:
    id: int
    angles: pd.DataFrame
    confidences: pd.DataFrame
    capacity: int
    mean_movement: float
    is_final: bool

    def get_last_angles(self) -> list[float]:
        if self.angles.empty:
            return [0.0] * len(PoseAngleJointNames)
        return self.angles.iloc[-1].tolist()

PoseStreamDataCallback = Callable[[PoseStreamData], None]
PoseStreamDataDict = dict[int, PoseStreamData]

class PoseStreamManager:
    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings
        num_players: int = settings.num_players

        self.processors: list[PoseStreamProcessor] = []
        self.result_queues: list[Queue] = []
        self.result_threads: list[Thread] = []
        for i in range(num_players):
            self.result_queues.append(Queue())
            self.processors.append(PoseStreamProcessor(settings, self.result_queues[i]))
            self.result_threads.append(Thread(target=self._handle_results, args=(self.result_queues[i],), daemon=True))

        self.output_callbacks: Set[PoseStreamDataCallback] = set()
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
        """Restart all processors and result threads when files change."""
        try:
            # Stop all processors
            for processor in self.processors:
                if processor.is_alive():
                    print(f"[PoseStream] Stopping processor {processor.pid}...")
                    processor.stop()
                    processor.join(timeout=2.0)
                    if processor.is_alive():
                        print(f"[PoseStream] Force terminating processor {processor.pid}...")
                        processor.terminate()
                        processor.join(timeout=1.0)

            # Clear old lists
            self.processors.clear()
            self.result_queues.clear()
            self.result_threads.clear()

            # Recreate processors, queues, and threads
            num_players = self.settings.num_players
            for i in range(num_players):
                queue = Queue()
                self.result_queues.append(queue)
                processor = PoseStreamProcessor(self.settings, queue)
                self.processors.append(processor)
                thread = Thread(target=self._handle_results, args=(queue,), daemon=True)
                self.result_threads.append(thread)

            # Start new processors and threads
            for processor in self.processors:
                processor.start()
            for thread in self.result_threads:
                thread.start()

            print("[PoseStream] All processors restarted successfully")

        except Exception as e:
            print(f"[PoseStream] Error restarting processors: {e}")

    def start(self) -> None:
        """Start all processors and result handler threads."""
        self.running = True
        for processor in self.processors:
            processor.start()
        for thread in self.result_threads:
            thread.start()

        self.hot_reloader.start_file_watcher()

    def stop(self) -> None:
        """Stop all processors and result handler threads."""
        self.running = False

        # Stop hot reloader
        self.hot_reloader.stop_file_watcher()

        # Stop all processors
        for processor in self.processors:
            processor.stop()
            processor.join(timeout=1.0)
            if processor.is_alive():
                processor.terminate()
                processor.join(timeout=1.0)

    def add_pose(self, pose) -> None:
        """Add pose to the appropriate processor's queue based on pose id."""
        try:
            pose_stream_input: PoseStreamInput = PoseStreamInput.from_pose(pose)
            # Distribute by id modulo number of processors
            processor_idx = pose_stream_input.id % len(self.processors)
            self.processors[processor_idx].add_pose(pose_stream_input)
        except Exception as e:
            print(f"[PoseStream] Error adding pose: {e}")

    def add_stream_callback(self, callback: PoseStreamDataCallback) -> None:
        """Register a callback to receive processed data."""
        self.output_callbacks.add(callback)

    def _handle_results(self, result_queue: Queue) -> None:
        """Handle results from a single processor's result queue in the main process."""
        while self.running:
            try:
                data: PoseStreamData = result_queue.get(timeout=0.1)
                for callback in self.output_callbacks:
                    try:
                        # this might need a lock if callbacks modify shared state
                        callback(data)
                    except Exception as e:
                        print(f"Error in callback: {e}")
            except:
                continue


class PoseStreamProcessor(Process):
    def __init__(self, settings: Settings, result_queue: Queue) -> None:
        super().__init__()

        self._stop_event = Event()
        self.pose_input_queue: Queue[PoseStreamInput] = Queue()

        # For sending results back to main process
        self.result_queue: Queue[PoseStreamData] = result_queue if result_queue else Queue()

        # Store settings values (not the settings object itself)
        self.buffer_capacity: int = settings.pose_stream_capacity
        self.resample_interval: str = f"{int(1.0 / settings.camera_fps * 1000)}ms"

        # Initialize buffers (will be recreated in child process)
        self.empty_df: pd.DataFrame = pd.DataFrame(columns=PoseAngleJointNames, dtype=float)
        self.angle_df: pd.DataFrame = self.empty_df.copy()
        self.score_df: pd.DataFrame = self.empty_df.copy()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while not self._stop_event.is_set():
            poses: list[PoseStreamInput] = []
            queue_empty = False
            while not queue_empty:
                try:
                    pose: PoseStreamInput = self.pose_input_queue.get(block=False)
                    if pose is not None:
                        poses.append(pose)
                except Empty:
                    queue_empty = True

            if not poses:
                sleep(0.01)
                continue

            try:
                self._process(poses)
            except Exception as e:
                print(f"Error processing poses: {e}")
                traceback.print_exc()

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

    def _process(self, poses: list[PoseStreamInput]) -> None:
        """ Process a pose and update the joint angle windows. """
        if not poses:
            return

        for pose in poses:
            if pose.is_removed:
                # reset buffers if pose is removed
                self.angle_df: pd.DataFrame = self.empty_df.copy()
                self.score_df: pd.DataFrame = self.empty_df.copy()
                self._notify_callbacks(PoseStreamData(pose.id, self.angle_df, self.score_df, self.buffer_capacity, 0.0, True))
                return
            if pose.angles is None:
                print(f"[PoseStreamProcessor] Pose {pose.id} has no angles, this should not happen")
                return

        angle_slice_df, conf_slice_df = self.get_data_frames_from_poses(poses)
        self.angle_df = pd.concat([self.angle_df, angle_slice_df])
        self.angle_df.sort_index(inplace=True)
        self.angle_df = self.angle_df.iloc[-self.buffer_capacity:]

        self.score_df = pd.concat([self.score_df, conf_slice_df])
        self.score_df.sort_index(inplace=True)
        self.score_df = self.score_df.iloc[-self.buffer_capacity:]

        # print(self.angle_df)
        try:
            interpolated: pd.DataFrame = self.angle_df.interpolate(method='time', limit_direction='both')#, limit=1)
        except Exception as e:
            print(f"Error interpolating angles: {e}")
            print(self.angle_df)

        smoothed: pd.DataFrame = PoseStreamProcessor.ewm_circular_mean(interpolated, span=7.0)
        mean_movement: float = PoseStreamProcessor.get_highest_movement(smoothed, self.score_df, threshold=0.0)

        # interpolated_score: pd.DataFrame = self.score_df.interpolate(method='time', limit_direction='both')#, limit=15)

        self._notify_callbacks(PoseStreamData(pose.id, smoothed, self.score_df, self.buffer_capacity, mean_movement, False))

    @ staticmethod
    def get_data_frames_from_poses(poses: list[PoseStreamInput]) ->tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert a list of PoseStreamInput objects to two DataFrames: one for angles and one for confidences.
        Args:
            poses: List of PoseStreamInput objects to process
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (angles_df, confidence_df)
        """

        if not poses or all(pose.angles is None for pose in poses):
            empty = pd.DataFrame(columns=PoseAngleJointNames, dtype=float)
            return empty, empty.copy()

        timestamps: list[pd.Timestamp] = [pose.time_stamp for pose in poses if pose.angles is not None]
        angle_data: list[np.ndarray] = [pose.angles.angles for pose in poses if pose.angles is not None]
        conf_data: list[np.ndarray] = [pose.angles.scores for pose in poses if pose.angles is not None]

        angles_df = pd.DataFrame(angle_data, index=timestamps, columns=PoseAngleJointNames, dtype=float)
        conf_df = pd.DataFrame(conf_data, index=timestamps, columns=PoseAngleJointNames, dtype=float)

        return angles_df, conf_df

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

    @staticmethod
    def get_mean_movement(angles: pd.DataFrame, confidences: pd.DataFrame, num_samples: int, threshold: float = 0.0) -> float:
        """Calculate the mean movement across all angles using only the last num_samples."""
        if angles.empty or confidences.empty:
            return 0.0

        # Use only the last num_samples rows
        angles_tail = angles.tail(num_samples)
        confidences_tail = confidences.tail(num_samples)

        # Calculate absolute differences between consecutive angles
        angle_diffs: pd.DataFrame = angles_tail.diff().abs()
        # Apply confidence mask
        angle_diffs *= confidences_tail

        # Calculate mean movement where confidence is above threshold
        valid_movement: pd.DataFrame = angle_diffs[confidences_tail > threshold]
        mean_val: float = valid_movement.mean().mean() if not valid_movement.empty else 0.0
        if pd.isna(mean_val):
            return 0.0
        return float(mean_val)

    @staticmethod
    def get_highest_movement(angles: pd.DataFrame, confidences: pd.DataFrame, threshold: float = 0.0) -> float:
        """Calculate the highest joint movement using only the last sample (difference between last two rows)."""
        if angles.empty or confidences.empty or len(angles) < 2:
            return 0.0

        # Use only the last two rows to get the most recent movement
        angles_tail = angles.tail(2)
        confidences_tail = confidences.tail(2)

        # Calculate absolute differences between the last two samples
        angle_diffs: pd.DataFrame = angles_tail.diff().abs().iloc[-1:]
        # Apply confidence mask
        angle_diffs *= confidences_tail.iloc[-1:]

        # Only consider movements where confidence is above threshold
        valid_mask = confidences_tail.iloc[-1:] > threshold
        angle_diffs = angle_diffs.where(valid_mask, 0.0)

        max_val = angle_diffs.max().max()
        if pd.isna(max_val):
            return 0.0
        return float(max_val)
