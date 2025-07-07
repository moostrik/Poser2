# Standard library imports
# from multiprocessing.synchronize import Event

import pickle
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from multiprocessing import Process, Queue, Event, cpu_count
from typing import Optional, Callable

# Third-party imports
import numpy as np
from numba import njit
import pandas as pd
import fastdtw

# Local application imports
from modules.pose.PoseWindowBuffer import PoseWindowData
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

PoseWindowDict = dict[int, PoseWindowData]

class CorrelationMethod(Enum):
    ANGULAR = "angular"

class PoseCorrelation:
    def __init__(self, id_1: int, id_2: int, joint_correlations: dict[str, float]) -> None:
        self.id_1 = id_1
        self.id_2 = id_2
        self.joint_correlations: dict[str, float] = joint_correlations
        self.similarity_score: float = float(np.mean(list(joint_correlations.values()))) if joint_correlations else 0.0

class PoseCorrelationBatch:
    def __init__(self) -> None:
        """Collection of DTW results from a single analysis run."""
        self._pair_correlations: list[PoseCorrelation] = []
        self._timestamp: pd.Timestamp = pd.Timestamp.now()
        self._similarity: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Check if the batch has no results."""
        return len(self._pair_correlations) == 0

    @property
    def count(self) -> int:
        """Return the number of windows in the batch."""
        return len(self._pair_correlations)

    @property
    def timestamp(self) -> pd.Timestamp:
        return self._timestamp

    @property
    def similarity(self) -> float:
        return self._similarity

    def add_result(self, result: PoseCorrelation) -> None:
        """Add a PoseCorrelation result to the batch."""
        self._pair_correlations.append(result)
        self._similarity = sum(r.similarity_score for r in self._pair_correlations) / len(self._pair_correlations)

    def get_most_similar_pair(self) -> Optional[PoseCorrelation]:
        """Return the pair with highest similarity score."""
        if not self._pair_correlations:
            return None
        return max(self._pair_correlations, key=lambda r: r.similarity_score)

@dataclass
class AnglePair:
    id_1: int
    id_2: int
    angles_1: pd.DataFrame
    angles_2: pd.DataFrame
    confidences_1: pd.DataFrame
    confidences_2: pd.DataFrame

def angular_cost(theta1: float, theta2: float):
    diff: float = theta1 - theta2
    # 1 - cos(diff) ∈ [0, 2], 0 if equal angles
    return 1.0 - np.cos(diff)

def dtw_angular_sakoe_chiba(x: np.ndarray, y: np.ndarray, band) -> float:
    n, m = len(x), len(y)
    dtw: np.ndarray = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n+1):
        j_start: int = max(1, i - band)
        j_end: int = min(m+1, i + band + 1)
        for j in range(j_start, j_end):
            cost: float = angular_cost(x[i-1], y[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            )
    return np.sqrt(dtw[n, m])

@njit
def angular_cost_njit(theta1: float, theta2: float):
    diff: float = theta1 - theta2
    # 1 - cos(diff) ∈ [0, 2], 0 if equal angles
    return 1.0 - np.cos(diff)

@njit
def dtw_angular_sakoe_chiba_njit(x: np.ndarray, y: np.ndarray, band) -> float:
    n, m = len(x), len(y)
    dtw: np.ndarray = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n+1):
        j_start: int = max(1, i - band)
        j_end: int = min(m+1, i + band + 1)
        for j in range(j_start, j_end):
            cost: float = angular_cost_njit(x[i-1], y[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            )
    return np.sqrt(dtw[n, m])

def ignore_keyboard_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class PoseAnalysis(Process):  # Changed from Thread to Process
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()

        # Use multiprocessing Queue for thread-safe communication
        self.input_queue: Queue = Queue(maxsize=240)
        self.result_queue: Queue = Queue(maxsize=240)

        self.analysis_interval: float = 1.0 / settings.analysis_rate_hz
        self.max_window_size: int = int(settings.pose_window_size * settings.camera_fps)
        self.analysis_window_size: int = min(int(settings.analysis_window_size * settings.camera_fps), self.max_window_size)

        self.align_tolerance = pd.Timedelta(f'{int(1000 / settings.camera_fps)}ms')
        self.maximum_nan_ratio: float = 0.15
        self.max_age: float = 1.0

        self.max_workers = min(cpu_count(), 10)

    def set_window(self, data: PoseWindowData) -> None:
        """Non-blocking method to send data to the analysis process."""
        try:
            # Serialize the data to avoid pickle issues
            serialized_data: bytes = pickle.dumps(data)
            self.input_queue.put(serialized_data, block=False)
        except:
            print(f"[{self.__class__.__name__}] Input queue is full, skipping window update for {data.window_id}")
            # Queue is full, skip this update
            pass

    def get_results(self) -> Optional[PoseCorrelationBatch]:
        """Non-blocking method to get results from the analysis process."""
        try:
            return self.result_queue.get(block=False)
        except:
            return None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        ignore_keyboard_interrupt()
        pose_windows: PoseWindowDict = {}
        hot_reloader = HotReloadMethods(self.__class__, False, reload_everything=False)
        process_pool = ProcessPoolExecutor(max_workers=self.max_workers, initializer=ignore_keyboard_interrupt)

        try:
            while not self._stop_event.is_set():
                loop_start: float = time.perf_counter()

                # Collect all pending window updates
                while True:
                    try:
                        serialized_data: bytes = self.input_queue.get(block=False)
                        data: PoseWindowData = pickle.loads(serialized_data)
                        pose_windows[data.window_id] = data
                    except:
                        break

                try:
                    pose_windows = self._prepare_windows(pose_windows)
                    angle_pairs: list[AnglePair] = self._generate_asof_angle_pairs(pose_windows, self.align_tolerance)

                    if angle_pairs:
                        start_time: float = time.perf_counter()
                        batch = PoseCorrelationBatch()

                        # Submit all pairs for analysis
                        future_to_pair: dict[Future, AnglePair] = {}
                        for pair in angle_pairs:
                            future: Future[PoseCorrelation | None] = process_pool.submit(self._analyse_pair, pair, self.maximum_nan_ratio)
                            future_to_pair[future] = pair

                        # Collect results
                        for future in as_completed(future_to_pair):
                            pair: AnglePair = future_to_pair[future]
                            try:
                                correlation: Optional[PoseCorrelation] = future.result()
                                if correlation:
                                    batch.add_result(correlation)
                            except Exception as e:
                                print(f"Analysis failed for pair {pair.id_1}-{pair.id_2}: {e}")

                        # Get most similar if we have results
                        if not batch.is_empty:
                            most_similar: Optional[PoseCorrelation] = batch.get_most_similar_pair()

                        analysis_duration: float = time.perf_counter() - start_time
                        print(f"Analysis completed in {analysis_duration:.2f} seconds, found {batch.count} pairs.")

                        # Send results back to main process
                        try:
                            self.result_queue.put(batch, block=False)
                        except:
                            # Queue is full, skip this result
                            pass

                except Exception as e:
                    print(f"Error during analysis: {e}")
                    continue

                if hot_reloader.file_changed:
                    print(f"[{self.__class__.__name__}] Reloading methods due to file change")
                    hot_reloader.reload_methods()

                elapsed: float = time.perf_counter() - loop_start
                sleep_time: float = self.analysis_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # except KeyboardInterrupt:
        #     print(f"[{self.__class__.__name__}] Received KeyboardInterrupt, shutting down...")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Unexpected error: {e}")
        finally:
            process_pool.shutdown(wait=False)
            hot_reloader.stop_file_watcher()

    def _prepare_windows(self, windows: PoseWindowDict) -> PoseWindowDict:
        """Filter windows based on time and length."""
        windows = self._filter_windows_by_time(windows, self.max_age)
        windows = self._filter_windows_by_length(windows, self.analysis_window_size)
        windows = self._trim_windows_to_length(windows, self.analysis_window_size)
        # windows = self._filter_windows_by_nan(windows, self.nan_ratio)
        # for data in windows.values():
        #     data.angles.index = pd.to_datetime(data.angles.index).round(self.round)
        #     data.confidences.index = pd.to_datetime(data.confidences.index).round(self.round)
        return windows


    @staticmethod
    def _analyse_pair(pair: AnglePair, maximum_nan_ratio: float) -> Optional[PoseCorrelation]:
        """Analyse a single pair - designed for thread pool execution."""
        joint_correlations: dict[str, float] = {}
        angles_columns: pd.Index[str] = pair.angles_1.select_dtypes(include=[np.number]).columns

        for column in angles_columns:
            if column not in pair.angles_2.columns:
                continue

            # Get angle sequences as NumPy arrays
            angles_1_nan: np.ndarray = pair.angles_1[column].values.astype(float)
            angles_2_nan: np.ndarray = pair.angles_2[column].values.astype(float)
            
            # Create mask for non-NaN values
            mask: np.ndarray = ~np.isnan(angles_1_nan) & ~np.isnan(angles_2_nan)
            
            # Apply mask to get valid angles
            angles_1: np.ndarray = angles_1_nan[mask]
            angles_2: np.ndarray = angles_2_nan[mask]

            # Calculate NaN ratio
            nan_ratio: float = 1.0 - (len(angles_1) / len(angles_1_nan))
            
            # Skip correlation if too many NaNs
            if nan_ratio > maximum_nan_ratio:
                joint_correlations[column] = 0.0
                continue

            # Calculate similarity
            try:
                method = CorrelationMethod.ANGULAR
                similarity: float = PoseAnalysis._compute_correlation(angles_1, angles_2, method=method)
                joint_correlations[column] = similarity
            except Exception as e:
                print(f"{column}: correlation failed - {e}")
                continue
        
        # Return correlation object if we have results
        if joint_correlations:
            return PoseCorrelation(
                id_1=pair.id_1,
                id_2=pair.id_2,
                joint_correlations=joint_correlations
            )
        return None

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


            print(len(angles1_overlap), len(angles2_overlap))

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

    @staticmethod
    def _generate_asof_angle_pairs(windows: PoseWindowDict, tolerance: pd.Timedelta) -> list[AnglePair]:
        """
        For each unique pair of PoseWindowData, align their angles and confidences DataFrames using merge_asof.
        Returns a list of AnglePair(id1, id2, angles1_aligned, angles2_aligned, confidences1_aligned, confidences2_aligned).
        """
        pairs: list[AnglePair] = []
        window_items: list[tuple[int, PoseWindowData]] = list(windows.items())

        for (id1, win1), (id2, win2) in combinations(window_items, 2):
            # Get the DataFrames
            angles_1: pd.DataFrame = win1.angles
            angles_2: pd.DataFrame = win2.angles
            confidences_1: pd.DataFrame = win1.confidences
            confidences_2: pd.DataFrame = win2.confidences

            # Align angles_2 to angles_1's timestamps
            angles_2_aligned: pd.DataFrame = pd.merge_asof(
                angles_1.reset_index(),
                angles_2.reset_index(),
                on='index',
                direction='nearest',
                tolerance=tolerance,
                suffixes=('_1', '')
            ).set_index('index')

            # Align confidences_2 to angles_1's timestamps (same alignment as angles)
            confidences_2_aligned: pd.DataFrame = pd.merge_asof(
                angles_1.reset_index()[['index']],  # Only need the index column for alignment
                confidences_2.reset_index(),
                on='index',
                direction='nearest',
                tolerance=tolerance,
                suffixes=('_1', '')
            ).set_index('index')

            # Only create pair if we have valid aligned data
            if len(angles_2_aligned) > 0 and len(confidences_2_aligned) > 0:
                P = AnglePair(
                        id_1=id1,
                        id_2=id2,
                        angles_1=angles_1,
                        angles_2=angles_2_aligned,
                        confidences_1=confidences_1,
                        confidences_2=confidences_2_aligned
                    )
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)
                pairs.append(P)

        return pairs

    @staticmethod
    def _compute_correlation(seq1: np.ndarray, seq2: np.ndarray, method: CorrelationMethod) -> float:
        """
        Compute correlation between two sequences using DTW with the specified method.

        Args:
            seq1: First sequence as 1D array of angles
            seq2: Second sequence as 1D array of angles
            method: Correlation method to use

        Returns:
            Similarity score between 0 and 1
        """
        # Map correlation methods to distance functions
        distance_functions = {
            CorrelationMethod.ANGULAR: lambda x, y: PoseAnalysis._angular_distance(x, y)  # Remove [0] indexing
        }

        # Select the distance function
        if method not in distance_functions:
            raise ValueError(f"Unknown correlation method: {method}")

        distance_func: Callable = distance_functions[method]

        # Compute DTW with the selected distance function

        t0: float = time.perf_counter()
        distance, path = fastdtw.fastdtw(
            seq1,  # Shape: (n_frames,) - just angles
            seq2,  # Shape: (n_frames,) - just angles
            # radius=10,
            dist=distance_func
        )
        # print(f"DTW distance computed in {time.perf_counter() - t0:.2f}s")

        # distance: float = dtw_angular_sakoe_chiba(seq1, seq2, band=5)


        # Normalize by path length (distance is already 0-1 from distance function)
        normalized_distance: float = (distance / len(seq1)) / np.pi

        # Convert distance to similarity (1 - distance)
        similarity: float = 1.0 - normalized_distance
        # print(f"DTW distance: {distance:.3f}, normalized: {normalized_distance:.3f}, similarity: {similarity:.3f}")
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

    @staticmethod
    def _angular_distance(angle_A, angle_B) -> float:
        """
        Calculate the shortest angular distance between two angles in radians,
        weighted by their confidence values and handling NaN values.

        Args:
            angle_A: First angle in radians
            confidence_A: Confidence value for first angle (0.0 to 1.0)
            angle_B: Second angle in radians
            confidence_B: Confidence value for second angle (0.0 to 1.0)

        Returns:
            Normalized weighted angular distance between 0 and 1
        """
        # Handle NaN values - return maximum distance if either angle is NaN
        # if not mask:
        #     return np.pi

        # Calculate the shortest angular distance using basic math operations
        diff: float = abs(angle_A - angle_B)
        angular_dist: float = min(diff, 2 * np.pi - diff)

        # Weight the distance by confidence - lower confidence increases distance
        # min_confidence: float = min(confidence_A, confidence_B)
        # confidence_weight: float = 1.0 - min_confidence

        # # Apply confidence weighting: low confidence -> higher distance
        # weighted_distance: float = angular_dist * (1.0 + confidence_weight)

        # Ensure result stays within [0, π] range
        return min(angular_dist, 1.0, np.pi)


