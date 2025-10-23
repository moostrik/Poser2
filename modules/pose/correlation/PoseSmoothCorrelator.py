# Standard library imports
# from multiprocessing.synchronize import Event

import signal
import time
import threading
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from multiprocessing import cpu_count
from typing import Optional, Callable

# Third-party imports
import numpy as np
from numba import njit
import pandas as pd

# Local application imports
from modules.pose.correlation.PairCorrelation import PairCorrelation, PairCorrelationBatch, PoseCorrelationBatchCallback
from modules.pose.PoseStream import PoseStreamData, PoseStreamDataDict
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class AnglePair:
    id_1: int
    id_2: int
    angles_1: pd.DataFrame
    angles_2: pd.DataFrame
    confidences_1: Optional[pd.DataFrame]
    confidences_2: Optional[pd.DataFrame]

@njit
def angular_cost(theta1: float, theta2: float) -> float:
    diff: float = abs(theta1 - theta2)
    return min(diff, 2 * np.pi - diff)

@njit
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

    return dtw[n, m]

@njit
def dtw_angular_sakoe_chiba_path(x: np.ndarray, y: np.ndarray, band) -> tuple[float, int]:
    n, m = len(x), len(y)
    dtw: np.ndarray = np.full((n+1, m+1), np.inf)
    # Track which direction we came from
    path_choices = np.zeros((n+1, m+1), dtype=np.int32)
    dtw[0, 0] = 0.0

    for i in range(1, n+1):
        j_start: int = max(1, i - band)
        j_end: int = min(m+1, i + band + 1)
        for j in range(j_start, j_end):
            cost: float = angular_cost(x[i-1], y[j-1])

            # Find minimum of three options
            min_prev = min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
            dtw[i, j] = cost + min_prev

            # Track which direction had minimum (0=diagonal, 1=vertical, 2=horizontal)
            if min_prev == dtw[i-1, j-1]:
                path_choices[i, j] = 0  # diagonal
            elif min_prev == dtw[i-1, j]:
                path_choices[i, j] = 1  # vertical
            else:
                path_choices[i, j] = 2  # horizontal

    # Trace back to find path length
    i, j = n, m
    path_length = 0
    while i > 0 or j > 0:
        path_length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choice = path_choices[i, j]
            if choice == 0:  # diagonal
                i -= 1
                j -= 1
            elif choice == 1:  # vertical
                i -= 1
            else:  # horizontal
                j -= 1

    return dtw[n, m], path_length

def ignore_keyboard_interrupt() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class DTWCorrelator():
    def __init__(self, settings: Settings) -> None:

        self.interval: float = 1.0 / settings.corr_rate_hz

        self.max_buffer_capacity: int = settings.pose_stream_capacity
        self.buffer_capacity: int = min(settings.corr_buffer_duration, self.max_buffer_capacity)
        self.stream_timeout: float = settings.corr_stream_timeout

        self.maximum_nan_ratio: float = settings.corr_max_nan_ratio
        self.dtw_band: int = settings.corr_dtw_band
        self.similarity_exponent: float = settings.corr_similarity_exp

        self._max_workers: int = max(min(cpu_count(), settings.corr_num_workers), 1)

        self._process_pool = ProcessPoolExecutor(max_workers=self._max_workers, initializer=ignore_keyboard_interrupt)
        self._correlation_thread = threading.Thread(target=self.run)
        self._stop_event = threading.Event()

        # INPUTS
        self._input_lock = threading.Lock()
        self._input_pose_streams: PoseStreamDataDict = {}

        # CALLBACKS
        self._callback_lock = threading.Lock()
        self._callbacks: set[PoseCorrelationBatchCallback] = set()

        # HOT RELOADER
        self.hot_reloader = HotReloadMethods(self.__class__, False, False)
        self.hot_reloader.add_file_changed_callback(self._reload_and_restart)

    def set_pose_stream(self, data: PoseStreamData) -> None:
        with self._input_lock:
            self._input_pose_streams[data.id] = data
        return

    def add_correlation_callback(self, callback: PoseCorrelationBatchCallback) -> None:
        """ Register a callback to receive the last correlation batch. """
        with self._callback_lock:
            self._callbacks.add(callback)

    def _notify_callbacks(self, batch: PairCorrelationBatch) -> None:
        """ Call all registered callbacks with the current batch. """
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    print(f"[{self.__class__.__name__}] Callback error: {e}")

    def start(self) -> None:
        self._correlation_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._correlation_thread.join()
        self._process_pool.shutdown(wait=True, cancel_futures=True)
        with self._callback_lock:
            self._callbacks.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            loop_start: float = time.perf_counter()

            with self._input_lock:
                pose_streams: PoseStreamDataDict = self._input_pose_streams.copy()

            # print(pose_streams)
            pose_streams = self._filter_streams_by_angles(pose_streams, 4) # use only arms
            pose_streams = self._filter_streams_by_time(pose_streams, self.stream_timeout)
            angle_pairs: list[AnglePair] = self._generate_naive_angle_pairs(pose_streams, self.buffer_capacity)

            if angle_pairs:
                future_to_pair: dict[Future, AnglePair] = {}
                try:
                    for pair in angle_pairs:
                        try:
                            future: Future[PairCorrelation | None] = self._process_pool.submit(
                                self._analyse_pair, pair, self.maximum_nan_ratio, self.dtw_band, self.similarity_exponent
                            )
                            future_to_pair[future] = pair
                        except BrokenProcessPool as bpe:
                            print(f"Failed to submit analysis for pair {pair.id_1}-{pair.id_2}: {bpe}")
                            self._stop_event.set()
                            break
                except BrokenProcessPool as bpe:
                    print(f"Process pool broken during submission: {bpe}")
                    self._stop_event.set()
                    break

                correlations: list[PairCorrelation] = []
                try:
                    for future in as_completed(future_to_pair):
                        pair: AnglePair = future_to_pair[future]
                        try:
                            correlation: Optional[PairCorrelation] = future.result()
                            if correlation:
                                correlations.append(correlation)
                        except Exception as e:
                            print(f"Analysis failed for pair {pair.id_1}-{pair.id_2}: {e}")
                except BrokenProcessPool as bpe:
                    print(f"Process pool broken during result collection: {bpe}")
                    self._stop_event.set()
                    break

                batch = PairCorrelationBatch(pair_correlations=correlations)
                self._notify_callbacks(batch)

            elapsed: float = time.perf_counter() - loop_start
            sleep_time: float = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _reload_and_restart(self) -> None:
        self._stop_event.set()
        self._correlation_thread.join()
        self._process_pool.shutdown(wait=True)

        self.hot_reloader.reload_methods()

        self._process_pool = ProcessPoolExecutor(max_workers=self._max_workers,initializer = ignore_keyboard_interrupt)
        self._stop_event.clear()
        self._correlation_thread = threading.Thread(target=self.run)
        self._correlation_thread.start()

    @staticmethod
    def _filter_streams_by_time(streams: PoseStreamDataDict, max_age_s: float = 2.0) -> PoseStreamDataDict:
        """Return only streams whose last timestamp is within max_age_s seconds from now."""
        now: pd.Timestamp = pd.Timestamp.now()
        filtered: PoseStreamDataDict = {}
        for id, data in streams.items():
            if not data.angles.empty:
                last_time = data.angles.index[-1]
                # print(last_time, (now - last_time).total_seconds())
                if (now - last_time).total_seconds() <= max_age_s:
                    filtered[id] = data
        return filtered

    @staticmethod
    def _filter_streams_by_angles(streams: PoseStreamDataDict, num_angles) -> PoseStreamDataDict:
        for key, data in streams.items():
            # angles: pd.DataFrame = data.angles.head(num_angles) # keep ony first angles
            # confidences: pd.DataFrame = data.confidences.head(num_angles) # keep ony first angles
            angles: pd.DataFrame = data.angles.iloc[:, :num_angles] # keep ony first angles
            confidences: pd.DataFrame = data.confidences.iloc[:, :num_angles] # keep ony first angles
            # print(angles)

            streams[key] = replace(
                data,
                angles=angles,
                confidences=confidences
            )
        return streams

    @staticmethod
    def _filter_streams_by_length(streams: PoseStreamDataDict, min_length: int = 20) -> PoseStreamDataDict:
        """Return only streams with at least min_length frames."""
        return {wid: data for wid, data in streams.items() if len(data.angles) >= min_length}

    @staticmethod
    def _filter_streams_by_nan(streams: PoseStreamDataDict, min_valid_ratio: float = 0.7) -> PoseStreamDataDict:
        """Return only streams where the ratio of non-NaN values is above min_valid_ratio."""
        filtered: PoseStreamDataDict = {}
        for wid, data in streams.items():
            total: int = data.angles.size
            valid: int = data.angles.count().sum()
            if total > 0 and (valid / total) >= min_valid_ratio:
                filtered[wid] = data
        return filtered

    @ staticmethod
    def _remove_nans_from_streams(streams: PoseStreamDataDict) -> PoseStreamDataDict:
        """Remove NaN values from angles and confidences DataFrames in each stream."""
        for key, data in streams.items():
            angles: pd.DataFrame = data.angles.dropna()
            confidences: pd.DataFrame = data.confidences.dropna()
            streams[key] = replace(
                data,
                angles=angles,
                confidences=confidences
            )

        return streams

    @staticmethod
    def _trim_streams_to_length(streams: PoseStreamDataDict, max_length: int ) -> PoseStreamDataDict:
        """ Trim each stream's DataFrames to the last max_length frames. """
        for key, data in streams.items():
            if len(data.angles) > max_length:
                angles: pd.DataFrame = data.angles.iloc[-max_length:]
                confidences: pd.DataFrame = data.confidences.iloc[-max_length:]
            streams[key] = replace(
                data,
                angles=angles,
                confidences=confidences
            )
        return streams

    @staticmethod
    def _generate_overlapping_angle_pairs(streams: PoseStreamDataDict) -> list[AnglePair]:
        """Generate all unique pairs of streams with overlapping time ranges."""
        angle_pairs: list[AnglePair] = []
        stream_items: list[tuple[int, PoseStreamData]] = list(streams.items())

        for (id1, win1), (id2, win2) in combinations(stream_items, 2):
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


            # print(len(angles1_overlap), len(angles2_overlap))

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
    def _generate_asof_angle_pairs(streams: PoseStreamDataDict, tolerance: pd.Timedelta) -> list[AnglePair]:
        """
        For each unique pair of PoseStream
        Data, align their angles and confidences DataFrames using merge_asof.
        Returns a list of AnglePair(id1, id2, angles1_aligned, angles2_aligned, confidences1_aligned, confidences2_aligned).
        """
        pairs: list[AnglePair] = []
        stream_items: list[tuple[int, PoseStreamData]] = list(streams.items())

        for (id1, win1), (id2, win2) in combinations(stream_items, 2):
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

        return pairs

    @staticmethod
    def _generate_naive_angle_pairs(streams: PoseStreamDataDict, max_length: int) -> list[AnglePair]:
        """
        Generate angle pairs from streams,
        Returns a list of AnglePair(id1, id2, angles1, angles2, confidences_1, confidences_2).
        """
        pairs: list[AnglePair] = []
        stream_items: list[tuple[int, PoseStreamData]] = list(streams.items())

        for (id1, win1), (id2, win2) in combinations(stream_items, 2):
            # Get the DataFrames
            angles_1: pd.DataFrame = win1.angles
            angles_2: pd.DataFrame = win2.angles
            confidences_1: pd.DataFrame = win1.confidences
            confidences_2: pd.DataFrame = win2.confidences

            # trim to max_length
            if len(angles_1) < max_length:
                continue
            if len(angles_2) < max_length:
                continue

            angles_1 = angles_1.iloc[-max_length:]
            angles_2 = angles_2.iloc[-max_length:]
            confidences_1 = confidences_1.iloc[-max_length:]
            confidences_2 = confidences_2.iloc[-max_length:]

            P = AnglePair(
                id_1=id1,
                id_2=id2,
                angles_1=angles_1.iloc[-max_length:],
                angles_2=angles_2.iloc[-max_length:],
                confidences_1=confidences_1.iloc[-max_length:],
                confidences_2=confidences_2.iloc[-max_length:],
            )
            pairs.append(P)

        return pairs

    @staticmethod
    def _analyse_pair(pair: AnglePair, maximum_nan_ratio: float, dtw_band: int, similarity_exponent: float) -> Optional[PairCorrelation]:
        """
        Analyse a single pair of angle sequences for all joints and compute their similarity.

        For each numeric joint/column in the provided AnglePair, this method:
        - Extracts the angle sequences from both streams and removes NaN values independently.
        - Calculates the ratio of NaN values for each sequence and skips the joint if either exceeds the maximum allowed ratio.
        - Computes the similarity between the two cleaned angle sequences using DTW with an angular cost, applying the specified Sakoe-Chiba band and similarity exponent.
        - Collects the similarity scores for all valid joints into a dictionary.

        Returns a PoseCorrelation object containing the per-joint similarity scores if any valid correlations are found, otherwise returns None.

        Args:
            pair (AnglePair): The pair of angle DataFrames and metadata to compare.
            maximum_nan_ratio (float): Maximum allowed ratio of NaN values per sequence.
            dtw_band (int): Sakoe-Chiba band width for DTW.
            similarity_exponent (float): Exponent for similarity emphasis (e.g., 2.0 for quadratic).

        Returns:
            Optional[PoseCorrelation]: Correlation results for the pair, or None if no valid joints.
        """

        joint_correlations: dict[str, float] = {}
        angles_columns: pd.Index[str] = pair.angles_1.select_dtypes(include=[np.number]).columns

        for column in angles_columns:
            if column not in pair.angles_2.columns:
                continue

            # Get angle sequences as NumPy arrays
            angles_1_nan: np.ndarray = pair.angles_1[column].values.astype(float)
            angles_2_nan: np.ndarray = pair.angles_2[column].values.astype(float)

            # Remove NaNs independently
            angles_1: np.ndarray = angles_1_nan[~np.isnan(angles_1_nan)]
            angles_2: np.ndarray = angles_2_nan[~np.isnan(angles_2_nan)]

            # Calculate NaN ratio for each sequence
            nan_ratio_1: float = 1.0 - (len(angles_1) / len(angles_1_nan)) if len(angles_1_nan) > 0 else 1.0
            nan_ratio_2: float = 1.0 - (len(angles_2) / len(angles_2_nan)) if len(angles_2_nan) > 0 else 1.0

            # Skip correlation if too many NaNs
            if nan_ratio_1 > maximum_nan_ratio or nan_ratio_2 > maximum_nan_ratio:
                joint_correlations[column] = 0.0
                continue

            # Calculate similarity
            try:
                similarity: float = DTWCorrelator._compute_correlation(angles_1, angles_2, dtw_band, similarity_exponent)
                joint_correlations[column] = similarity
            except Exception as e:
                print(f"{column}: correlation failed - {e}")
                continue

        if joint_correlations:
            return PairCorrelation.from_ids(
                id_1=pair.id_1,
                id_2=pair.id_2,
                joint_correlations=joint_correlations
            )
        return None

    @staticmethod
    def _compute_correlation(seq1: np.ndarray, seq2: np.ndarray, band: int = 10, similarity_exponent: float = 2.0) -> float:
        """
        Compute the similarity between two angle sequences using DTW with an angular cost.

        - Uses dtw_angular_sakoe_chiba_path to calculate the DTW distance and path length between the sequences.
        - The distance is normalized by the path length and π, so 0 means identical angles, 1 means opposite.
        - The similarity is then calculated as (1 - normalized_distance) raised to the given 'exponent' power.
          This allows quadratic (2.0), cubic (3.0), or other power emphasis on high similarity.
        - The result is clipped to [0.0, 1.0].

        Args:
            seq1 (np.ndarray): First angle sequence (in radians).
            seq2 (np.ndarray): Second angle sequence (in radians).
            band (int): Sakoe-Chiba band width for DTW.
            similarity_exponent (float): Power applied to emphasize high similarity.

        Returns:
            float: Similarity score in [0.0, 1.0].
        """
        distance, path_length = dtw_angular_sakoe_chiba_path(seq1, seq2, band=band)
        normalized_distance: float = (distance / path_length) / np.pi
        similarity: float = (1.0 - normalized_distance)**similarity_exponent
        return max(0.0, min(1.0, similarity))
