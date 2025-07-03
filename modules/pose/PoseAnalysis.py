# Standard library imports
import time
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from threading import Event, Thread, Lock
from typing import Optional, Callable

# Third-party imports
import numpy as np
import pandas as pd
import fastdtw

# Local application imports
from modules.pose.PoseDefinitions import Keypoint
from modules.pose.PoseWindowBuffer import PoseWindowData
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

PoseWindowDict = dict[int, PoseWindowData]

class CorrelationMethod(Enum):
    DTW = "dtw"
    PEARSON = "pearson"
    COSINE = "cosine"
    MSE = "mse"
    CROSS_CORRELATION = "cross_correlation"

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
    confidences_1: Optional[pd.DataFrame]
    confidences_2: Optional[pd.DataFrame]


class PoseAnalysis(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()
        self.data_lock = Lock()
        self.pose_windows: PoseWindowDict = {}

        self.analysis_interval: float = 1.0 / settings.analysis_rate_hz
        self.max_window_size: int = int(settings.pose_window_size * settings.camera_fps)
        self.analysis_window_size: int = min(int(settings.analysis_window_size * settings.camera_fps), self.max_window_size)

        self.align_tolerance = pd.Timedelta(f'{int(1000 / settings.camera_fps)}ms')  # Round to nearest 45ms
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
                angle_pairs: list[AnglePair] = self._generate_asof_angle_pairs(pose_windows, self.align_tolerance)
                self._analyse(angle_pairs)
            except Exception as e:
                print(f"Error during analysis: {e}")
                continue

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

    def _analyse(self, angle_pairs: list[AnglePair]) -> PoseCorrelationBatch:
        batch = PoseCorrelationBatch()

        for pair in angle_pairs:
            joint_correlations: dict[str, float] = {}

            # Get columns from angles_1 that are numeric and find corresponding _2 columns in angles_2
            angles_columns: pd.Index[str] = pair.angles_1.select_dtypes(include=[np.number]).columns

            for column in angles_columns:
                # Check if corresponding _2 column exists in angles_2
                if column not in pair.angles_2.columns:
                    print(f"  Column {column} not found in angles_2")
                    continue

                # Get the angle sequences for this column (before dropna)
                seq1_raw: pd.Series = pair.angles_1[column]
                seq2_raw: pd.Series = pair.angles_2[column]

                # Check NaN ratio before dropping NaN values
                seq1_valid_ratio: float = seq1_raw.notna().sum() / len(seq1_raw) if len(seq1_raw) > 0.0 else 0.0
                seq2_valid_ratio: float = seq2_raw.notna().sum() / len(seq2_raw) if len(seq2_raw) > 0.0 else 0.0

                # Skip if either sequence has too many NaN values
                if seq1_valid_ratio < self.nan_ratio or seq2_valid_ratio < self.nan_ratio:
                    print(f"  Column {column}: Skipping due to low valid data ratio (seq1: {seq1_valid_ratio:.2f}, seq2: {seq2_valid_ratio:.2f})")
                    continue

                # Now get the clean sequences
                seq1: np.ndarray = np.array(seq1_raw.dropna().values, dtype=float)
                seq2: np.ndarray = np.array(seq2_raw.dropna().values, dtype=float)

                # Compute correlation using specified method
                try:
                    method = CorrelationMethod.DTW
                    similarity: float = PoseAnalysis._compute_correlation(seq1, seq2, method=method)
                    joint_correlations[column] = similarity
                    # print(f"  {column}: {method} similarity = {similarity:.3f}")
                except Exception as e:
                    print(f"  {column}: {method.value} failed - {e}")
                    continue

            # Create PoseCorrelation if we have valid results
            if joint_correlations:
                correlation = PoseCorrelation(
                    id_1=pair.id_1,
                    id_2=pair.id_2,
                    joint_correlations=joint_correlations
                )
                batch.add_result(correlation)
                # print(f"Overall similarity between {pair.id_1} and {pair.id_2}: {correlation.similarity_score:.3f}")
            else:
                print(f"No valid results for {pair.id_1} and {pair.id_2}")

        if not batch.is_empty:
            most_similar: Optional[PoseCorrelation] = batch.get_most_similar_pair()
            if most_similar:
                print(f"Most similar pair: {most_similar.id_1} ↔ {most_similar.id_2} (similarity: {most_similar.similarity_score:.3f})")
                pass

        return batch


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
        For each unique pair of PoseWindowData, align their angles DataFrames using merge_asof.
        Returns a list of AnglePair(id1, id2, angles1_aligned, angles2_aligned).
        """
        pairs: list[AnglePair] = []
        window_items: list[tuple[int, PoseWindowData]] = list(windows.items())
        for (id1, win1), (id2, win2) in combinations(window_items, 2):
            # Sort indices for merge_asof
            angles_1: pd.DataFrame = win1.angles
            angles_2: pd.DataFrame = win2.angles
            # Align win2 to win1's timestamps

            angles_2_aligned: pd.DataFrame = pd.merge_asof(
                angles_1.reset_index(),
                angles_2.reset_index(),
                on='index',
                direction='nearest',
                tolerance=tolerance,
                suffixes=('_1', '')
            ).set_index('index')

            if len(angles_2_aligned) > 0:
                pairs.append(
                    AnglePair(
                        id_1=id1,
                        id_2=id2,
                        angles_1=angles_1,
                        angles_2=angles_2_aligned,
                        confidences_1=None,
                        confidences_2=None
                    )
            )
        return pairs

    @staticmethod
    def _compute_correlation(seq1: np.ndarray, seq2: np.ndarray, method: CorrelationMethod = CorrelationMethod.DTW) -> float:
        """
        Compute correlation between two angle sequences using specified method.

        Args:
            seq1: First angle sequence in radians (numpy array, NaN-free)
            seq2: Second angle sequence in radians (numpy array, NaN-free)
            method: Correlation method from CorrelationMethod enum

        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        if method == CorrelationMethod.DTW:
            return PoseAnalysis._compute_dtw_similarity(seq1, seq2)
        elif method == CorrelationMethod.PEARSON:
            return PoseAnalysis._compute_pearson_similarity(seq1, seq2)
        elif method == CorrelationMethod.COSINE:
            return PoseAnalysis._compute_cosine_similarity(seq1, seq2)
        elif method == CorrelationMethod.MSE:
            return PoseAnalysis._compute_mse_similarity(seq1, seq2)
        elif method == CorrelationMethod.CROSS_CORRELATION:
            return PoseAnalysis._compute_cross_correlation_similarity(seq1, seq2)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    @staticmethod
    def _compute_dtw_similarity(seq1: np.ndarray, seq2: np.ndarray, radius: int = 3) -> float:
        """DTW similarity - handles different length sequences"""
        # Your existing DTW code
        def angular_distance(x, y):
            """Calculate the shortest angular distance between two angles in radians."""
            diff = np.abs(x - y)
            return np.minimum(diff, 2 * np.pi - diff)

        distance, path = fastdtw.fastdtw(
            seq1.reshape(-1, 1),  # Reshape to 2D for DTW
            seq2.reshape(-1, 1),  # Reshape to 2D for DTW
            radius=radius,
            dist=lambda x, y: angular_distance(x[0], y[0])
        )

        # Normalize by path length AND by π to get [0,1] range
        normalized_distance = distance / (len(path) * np.pi)

        # Convert distance to similarity (1 - distance)
        similarity = 1.0 - normalized_distance

        return similarity

    @staticmethod
    def _compute_pearson_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Pearson correlation - requires same length sequences"""
        min_len = min(len(seq1), len(seq2))
        seq1_trim = seq1[:min_len]
        seq2_trim = seq2[:min_len]

        correlation = np.corrcoef(seq1_trim, seq2_trim)[0, 1]
        return (correlation + 1) / 2  # Convert [-1,1] to [0,1]

    @staticmethod
    def _compute_cosine_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Cosine similarity - requires same length sequences"""
        min_len = min(len(seq1), len(seq2))
        seq1_trim = seq1[:min_len]
        seq2_trim = seq2[:min_len]

        dot_product = np.dot(seq1_trim, seq2_trim)
        norm_product = np.linalg.norm(seq1_trim) * np.linalg.norm(seq2_trim)

        if norm_product == 0:
            return 0.0

        cosine_sim = dot_product / norm_product
        return (cosine_sim + 1) / 2  # Convert [-1,1] to [0,1]

    @staticmethod
    def _compute_mse_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """MSE-based similarity with angular distance - requires same length"""
        min_len = min(len(seq1), len(seq2))
        seq1_trim = seq1[:min_len]
        seq2_trim = seq2[:min_len]

        # Use angular distance for MSE
        angular_diffs = np.abs(seq1_trim - seq2_trim)
        angular_diffs = np.minimum(angular_diffs, 2 * np.pi - angular_diffs)

        mse = np.mean(angular_diffs ** 2)
        return np.exp(-mse)  # Convert to similarity [0,1]

    @staticmethod
    def _compute_cross_correlation_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Cross-correlation similarity - handles different lengths"""
        correlation = np.correlate(seq1, seq2, mode='full')
        max_correlation = np.max(correlation)

        # Normalize by sequence lengths
        normalization = np.sqrt(np.sum(seq1**2) * np.sum(seq2**2))
        if normalization == 0:
            return 0.0

        return max_correlation / normalization