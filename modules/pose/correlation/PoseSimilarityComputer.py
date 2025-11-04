# Standard library imports
from itertools import combinations
import threading
import time
import traceback
from typing import Optional

# Third-party imports

# Pose imports
from ..features.PoseAngles import PoseAngleData
from ..features.PoseAngleSimilarity import PoseAngleSimilarityData , PoseSimilarityBatch , PoseSimilarityBatchCallback
from ..Pose import PoseDict

# Local application imports
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


class PoseSimilarityComputer:
    """Computes pairwise pose similarities in a background thread.

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
        self._output_data: Optional[PoseSimilarityBatch ] = None
        self._callback_lock = threading.Lock()
        self._callbacks: set[PoseSimilarityBatchCallback] = set()

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
        while not self._stop_event.is_set():
            self._update_event.wait()
            if self._stop_event.is_set():
                break
            self._update_event.clear()

            try:
                start_time: float = time.perf_counter()

                # Get input data
                with self._input_lock:
                    poses: PoseDict = self._input_poses

                # Process correlations
                batch: PoseSimilarityBatch  = self._evaluate_pose_similarity(poses)

                # Store and notify
                with self._output_lock:
                    self._output_data = batch
                self._notify_callbacks(batch)

                elapsed_time: float = time.perf_counter() - start_time
                # print(f"PoseCorrelator: Processed {len(batch.pair_correlations)} correlations in {elapsed_time:.3f}s")

            except Exception as e:
                print(f"PoseCorrelator: Processing error: {e}")
                traceback.print_exc()

    def _evaluate_pose_similarity(self, poses: PoseDict) -> PoseSimilarityBatch :
        """Process all pose pairs and compute their correlations."""
        # Extract angle data from actively tracked poses
        angle_data: dict[int, PoseAngleData] = {
            tracklet_id: pose.angle_data
            for tracklet_id, pose in poses.items()
            if pose.tracklet.is_being_tracked
        }

        if len(angle_data) < 2:
            return PoseSimilarityBatch(pair_correlations=[])

        # Compute correlations for each pair
        similarities: list[PoseAngleSimilarityData] = []
        for (id1, angles_1), (id2, angles_2) in combinations(angle_data.items(), 2):
            # Compute similarity scores
            similarity_data: PoseAngleData = angles_1.similarity(angles_2, self.similarity_exponent)

            # Only include pairs with at least one valid joint
            if similarity_data.any_valid:
                # Normalize pair_id ordering
                pair_id = (id1, id2) if id1 <= id2 else (id2, id1)

                similarities.append(PoseAngleSimilarityData(
                    pair_id=pair_id,
                    values=similarity_data.values,
                    scores=similarity_data.scores
                ))

        return PoseSimilarityBatch(pair_correlations=similarities)

    def add_poses(self, poses: PoseDict) -> None:
        """Update input poses and trigger correlation processing.

        Args:
            poses: Dictionary mapping tracklet IDs to Pose objects
        """
        with self._input_lock:
            self._input_poses = poses
            self._update_event.set()

    def get_output_data(self) -> Optional[PoseSimilarityBatch ]:
        """Get the latest computed correlation batch."""
        with self._output_lock:
            return self._output_data

    def add_correlation_callback(self, callback: PoseSimilarityBatchCallback) -> None:
        """Register a callback to receive correlation batch updates.

        Args:
            callback: Function to call with PairCorrelationBatch
        """
        with self._callback_lock:
            self._callbacks.add(callback)

    def _notify_callbacks(self, batch: PoseSimilarityBatch ) -> None:
        """Call all registered callbacks with the current batch."""
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    print(f"PoseCorrelator: Callback error: {e}")
                    traceback.print_exc()
