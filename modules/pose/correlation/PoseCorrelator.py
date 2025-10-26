# Standard library imports
import signal
import time
import threading
from dataclasses import dataclass
from itertools import combinations
import traceback
from typing import Optional

# Third-party imports
import numpy as np

# Local application imports
from modules.pose.Pose import Pose, PosePointData, PoseDict
from modules.pose.correlation.PairCorrelation import PairCorrelation, PairCorrelationBatch, PairCorrelationBatchCallback
from modules.pose.PoseStream import PoseStreamData, PoseStreamDataDict
from modules.Settings import Settings

from modules.pose.features.PoseAngles import AngleJoint
from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class AnglePair:
    """Pair of poses with their angle data for correlation analysis."""
    id_1: int
    id_2: int
    angles_1: dict[AngleJoint, float]
    angles_2: dict[AngleJoint, float]

class PoseCorrelator:
    """Computes pairwise pose correlations in a background thread.

    Processes pose angle data from active tracklets and computes similarity metrics
    for all pairs. Results are published via callbacks to registered listeners.
    """

    def __init__(self, settings: Settings) -> None:
        self.similarity_exponent: float = settings.corr_similarity_exp

        self._correlation_thread = threading.Thread(target=self.run, daemon=True)
        self._stop_event = threading.Event()

        # INPUTS - Just store the latest data with a lock
        self._input_lock = threading.Lock()
        self._input_poses: PoseDict = {}
        self._update_event = threading.Event()

        # OUTPUT AND CALLBACKS
        self._output_lock = threading.Lock()
        self._output_data: Optional[PairCorrelationBatch] = None
        self._callback_lock = threading.Lock()
        self._callbacks: set[PairCorrelationBatchCallback] = set()

        # HOT RELOADER
        self.hot_reloader = HotReloadMethods(self.__class__)

    def start(self) -> None:
        """Start the correlation processing thread."""
        self._correlation_thread.start()

    def stop(self) -> None:
        """Stop the correlation processing thread and clear callbacks."""
        self._stop_event.set()
        self._update_event.set()  # Wake the thread so it can see stop_event
        with self._callback_lock:
            self._callbacks.clear()

    def run(self) -> None:
        """Main correlation processing loop (runs in background thread)."""
        while True:
            self._update_event.wait()

            if self._stop_event.is_set():
                break

            self._update_event.clear()

            start_time: float = time.perf_counter()

            with self._input_lock:
                poses: PoseDict = self._input_poses

            angle_data: dict[int, dict[AngleJoint, float]] = self._extract_angles_from_poses(poses)
            angle_pairs: list[AnglePair] = self._generate_angle_pairs(angle_data)

            correlations: list[PairCorrelation] = []

            for pair in angle_pairs:
                try:
                    correlation: Optional[PairCorrelation] = self._analyse_pair(pair, self.similarity_exponent)
                    if correlation:
                        correlations.append(correlation)
                except Exception as e:
                    print(f"PoseCorrelator: Analysis failed for pair {pair.id_1}-{pair.id_2}: {e}")

            end_time: float = time.perf_counter()
            elapsed_time: float = end_time - start_time

            batch = PairCorrelationBatch(pair_correlations=correlations)
            with self._output_lock:
                self._output_data = batch
            self._notify_callbacks(batch)

    def add_poses(self, poses: PoseDict) -> None:
        """Update input poses and trigger correlation processing.

        Args:
            poses: Dictionary mapping tracklet IDs to Pose objects
        """
        with self._input_lock:
            self._input_poses = poses
            self._update_event.set()

    def get_output_data(self) -> Optional[PairCorrelationBatch]:
        """Get the latest computed correlation batch."""
        with self._output_lock:
            return self._output_data

    def add_correlation_callback(self, callback: PairCorrelationBatchCallback) -> None:
        """Register a callback to receive correlation batch updates.

        Args:
            callback: Function to call with PairCorrelationBatch
        """
        with self._callback_lock:
            self._callbacks.add(callback)

    def _notify_callbacks(self, batch: PairCorrelationBatch) -> None:
        """Call all registered callbacks with the current batch."""
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    print(f"PoseCorrelator: Callback error: {e}")
                    traceback.print_exc()

    @staticmethod
    def _extract_angles_from_poses(poses: PoseDict) -> dict[int, dict[AngleJoint, float]]:
        """Extract angle data from active poses.

        Only processes poses with active tracklets. Uses PoseAngleData.to_dict()
        to get all joint angles (includes NaN for occluded/missing joints).

        Args:
            poses: Dictionary mapping tracklet IDs to Pose objects

        Returns:
            Dictionary mapping tracklet IDs to their angle dictionaries
        """

        angle_dict: dict[int, dict[AngleJoint, float]] = {
            tracklet_id: pose.angle_data.to_dict()
            for tracklet_id, pose in poses.items()
            if pose.tracklet.is_active
        }
        # print(f"PoseCorrelator: Extracted angles for {len(angle_dict)} active poses.")

        return angle_dict


    @staticmethod
    def _generate_angle_pairs(angle_data: dict[int, dict[AngleJoint, float]]) -> list[AnglePair]:
        """Generate all pairwise combinations of angle data.

        Args:
            angle_data: Dictionary mapping tracklet IDs to angle dictionaries

        Returns:
            List of AnglePair objects for correlation analysis
        """
        pairs: list[AnglePair] = []
        data_items: list[tuple[int, dict[AngleJoint, float]]] = list(angle_data.items())

        for (id1, angles_1), (id2, angles_2) in combinations(data_items, 2):
            pair = AnglePair(
                id_1=id1,
                id_2=id2,
                angles_1=angles_1,
                angles_2=angles_2,
            )
            pairs.append(pair)

        return pairs

    @staticmethod
    def _analyse_pair(pair: AnglePair, similarity_exponent: float) -> Optional[PairCorrelation]:
        """Compute similarity scores for all joints in a pose pair.

        For each joint:
        - If either angle is NaN, stores NaN (joint not available for comparison)
        - Otherwise, computes angular difference and similarity score
        - Similarity = (1 - normalized_diff)^similarity_exponent

        The PairCorrelation object automatically filters NaN values when computing
        statistics (mean, geometric_mean, etc.), so NaN joints don't affect results.

        Args:
            pair: AnglePair containing two poses' angle data
            similarity_exponent: Exponent for similarity emphasis (e.g., 2.0 for quadratic)

        Returns:
            PairCorrelation with per-joint similarities, or None if no correlations computed
        """
        correlations: dict[str, float] = {}

        for angle_joint in AngleJoint:
            # Get angles from both poses (may be NaN)
            angle_1: float = pair.angles_1.get(angle_joint, np.nan)
            angle_2: float = pair.angles_2.get(angle_joint, np.nan)

            # If either angle is NaN, mark correlation as NaN
            # PairCorrelation will filter these out when computing stats
            if np.isnan(angle_1) or np.isnan(angle_2):
                correlations[angle_joint.name] = np.nan
                continue

            # Calculate angular difference (accounting for circular nature)
            diff: float = abs(angle_1 - angle_2)

            # Normalize to [0, π] range (shortest angular distance)
            if diff > np.pi:
                diff = 2 * np.pi - diff

            # Normalize by π: 0 = identical, 1 = opposite
            normalized_diff: float = diff / np.pi

            # Calculate similarity with exponent for emphasis
            similarity: float = (1.0 - normalized_diff) ** similarity_exponent
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

            correlations[angle_joint.name] = similarity

        # Return PairCorrelation if we have any correlations (even if all NaN)
        if correlations:
            return PairCorrelation.from_ids(
                id_1=pair.id_1,
                id_2=pair.id_2,
                correlations=correlations
            )
        return None
