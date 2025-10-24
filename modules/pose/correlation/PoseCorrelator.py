# Standard library imports
import signal
import time
import threading
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

# Third-party imports
import numpy as np

# Local application imports
from modules.pose.correlation.PairCorrelation import PairCorrelation, PairCorrelationBatch, PoseCorrelationBatchCallback
from modules.pose.PoseStream import PoseStreamData, PoseStreamDataDict
from modules.Settings import Settings

from modules.pose.features.PoseAngles import AngleJoint
from modules.utils.HotReloadMethods import HotReloadMethods


input_data = dict[int, dict[AngleJoint, float]]

@dataclass
class AnglePair:
    id_1: int
    id_2: int
    angles_1: dict[AngleJoint, float]
    angles_2: dict[AngleJoint, float]

class PoseCorrelator():
    def __init__(self, settings: Settings) -> None:

        self.similarity_exponent: float = settings.corr_similarity_exp

        self._correlation_thread = threading.Thread(target=self.run)
        self._stop_event = threading.Event()

        # INPUTS - Just store the latest data with a lock
        self._input_lock = threading.Lock()
        self._input_data: Optional[input_data] = None
        self._new_data_available = threading.Event()

        # OUTPUT AND CALLBACKS
        self._output_lock = threading.Lock()
        self._output_data: Optional[PairCorrelationBatch] = None
        self._callback_lock = threading.Lock()
        self._callbacks: set[PoseCorrelationBatchCallback] = set()

        # HOT RELOADER
        self.hot_reloader = HotReloadMethods(self.__class__, False, False)
        self.hot_reloader.add_file_changed_callback(self._reload_and_restart)

    def start(self) -> None:
        self._correlation_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._correlation_thread.join()
        with self._callback_lock:
            self._callbacks.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            # Wait for new data with timeout
            if not self._new_data_available.wait(timeout=0.1):
                continue

            # Get the latest data
            with self._input_lock:
                data: input_data | None = self._input_data
                self._new_data_available.clear()

            if data is None:
                continue

            start_time: float = time.perf_counter()

            angle_pairs: list[AnglePair] = self._generate_angle_pairs(data)

            correlations: list[PairCorrelation] = []
            for pair in angle_pairs:
                try:
                    correlation: Optional[PairCorrelation] = self._analyse_pair(pair, self.similarity_exponent)
                    if correlation:
                        correlations.append(correlation)
                except Exception as e:
                    print(f"PoseSmoothCorrelatorThread: Analysis failed for pair {pair.id_1}-{pair.id_2}: {e}")

            end_time: float = time.perf_counter()
            elapsed_time: float = end_time - start_time
            # print(f"PoseSmoothCorrelatorThread: Processed batch in {elapsed_time:.4f} seconds, 28 would be {elapsed_time / len(angle_pairs) * 28:.4f} seconds" if angle_pairs else "No pairs to process.")

            batch = PairCorrelationBatch(pair_correlations=correlations)
            with self._output_lock:
                self._output_data = batch
            self._notify_callbacks(batch)

    def set_pose_stream(self, data: PoseStreamData) -> None:
        """Add new pose data. Replaces any previous unprocessed data."""
        # print(data.get_last_angles())
        pass

        # with self._input_lock:
        #     self._input_data = data
        #     self._new_data_available.set()

    def get_output_data(self) -> Optional[PairCorrelationBatch]:
        """Get the latest computed correlation batch."""
        with self._output_lock:
            return self._output_data

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
                    print(f"PoseSmoothCorrelatorThread: [{self.__class__.__name__}] Callback error: {e}")

    def _reload_and_restart(self) -> None:
        self._stop_event.set()
        self._correlation_thread.join()

        self.hot_reloader.reload_methods()

        self._stop_event.clear()
        self._correlation_thread = threading.Thread(target=self.run)
        self._correlation_thread.start()

    @staticmethod
    def _generate_angle_pairs(data: input_data) -> list[AnglePair]:
        """
        Generate angle pairs from current angle data.
        Returns a list of AnglePair(id1, id2, angles1, angles2).
        """
        pairs: list[AnglePair] = []
        data_items: list[tuple[int, dict[AngleJoint, float]]] = list(data.items())

        for (id1, angles_1), (id2, angles_2) in combinations(data_items, 2):
            P = AnglePair(
                id_1=id1,
                id_2=id2,
                angles_1=angles_1,
                angles_2=angles_2,
            )
            pairs.append(P)

        return pairs

    @staticmethod
    def _analyse_pair(pair: AnglePair, similarity_exponent: float) -> Optional[PairCorrelation]:
        """
        Analyse a single pair of current poses for all angle joints and compute their similarity.

        For each AngleJoint in the provided AnglePair, this method:
        - Extracts the angle values from both poses
        - Skips the joint if either angle is NaN
        - Computes the angular difference between the two angles
        - Calculates similarity as (1 - normalized_difference)^similarity_exponent
        - Collects the similarity scores for all valid joints into a dictionary

        Returns a PairCorrelation object containing the per-joint similarity scores if any valid correlations are found, otherwise returns None.

        Args:
            pair (AnglePair): The pair of angle dictionaries to compare.
            similarity_exponent (float): Exponent for similarity emphasis (e.g., 2.0 for quadratic).

        Returns:
            Optional[PairCorrelation]: Correlation results for the pair, or None if no valid joints.
        """

        correlations: dict[str, float] = {}

        # Iterate through all AngleJoint types
        for angle_joint in AngleJoint:
            # Get angle values for both poses
            angle_1: float = pair.angles_1.get(angle_joint, np.nan)
            angle_2: float = pair.angles_2.get(angle_joint, np.nan)

            # Skip if either angle is NaN
            if np.isnan(angle_1) or np.isnan(angle_2):
                correlations[angle_joint.name] = 0.0
                continue

            # Calculate angular difference (accounting for circular nature of angles)
            diff: float = abs(angle_1 - angle_2)
            # Normalize to [0, π] range (shortest angular distance)
            if diff > np.pi:
                diff = 2 * np.pi - diff

            # Normalize by π so that 0 = identical, 1 = opposite
            normalized_diff: float = diff / np.pi

            # Calculate similarity
            similarity: float = (1.0 - normalized_diff) ** similarity_exponent
            similarity = max(0.0, min(1.0, similarity))

            correlations[angle_joint.name] = similarity

        if correlations:
            return PairCorrelation.from_ids(
                id_1=pair.id_1,
                id_2=pair.id_2,
                correlations=correlations
            )
        return None
