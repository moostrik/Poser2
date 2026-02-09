# Standard library imports
from dataclasses import dataclass
from enum import IntEnum
import threading
import time
import traceback
from typing import Optional, TYPE_CHECKING, cast
import warnings

# Third-party imports
import numpy as np

# Pose imports
from modules.ConfigBase import ConfigBase, config_field
from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.nodes.windows.WindowNode import FeatureWindow
from modules.pose.features.base.NormalizedScalarFeature import AggregationMethod, NormalizedScalarFeature
from modules.pose.features.Similarity import configure_similarity, Similarity
from modules.pose.features.LeaderScore import configure_leader_score, LeaderScore
from modules.utils.PerformanceTimer import PerformanceTimer

from modules.utils.HotReloadMethods import HotReloadMethods

# Type alias for window dictionary (track_id -> FeatureWindow)
WindowDict = dict[int, FeatureWindow]


class _JointAggregator(NormalizedScalarFeature):
    """Helper class for aggregating per-joint similarities into scalars.

    This is a minimal wrapper around NormalizedScalarFeature that allows us
    to reuse all aggregation methods (mean, harmonic mean, etc.) on arbitrary
    length arrays without needing a specific feature class.
    """
    _joint_enum: type[IntEnum] | None = None

    @classmethod
    def enum(cls) -> type[IntEnum]:
        if cls._joint_enum is None:
            raise RuntimeError("_JointAggregator not configured")
        return cls._joint_enum

    @classmethod
    def configure(cls, num_joints: int) -> None:
        """Configure the aggregator with the number of joints."""
        if cls._joint_enum is None:
            cls._joint_enum = cast(type[IntEnum], IntEnum("JointIndex", {f"J{i}": i for i in range(num_joints)}))


@dataclass
class WindowSimilarityConfig(ConfigBase):
    """Configuration for WindowSimilarity."""
    max_poses: int = config_field(3, min=1, max=10, description="Maximum number of tracked poses")
    window_length: int = config_field(30, min=1, max=300, description="Number of frames to compare")
    method: AggregationMethod = AggregationMethod.HARMONIC_MEAN
    exponent: float = config_field(3.5, min=0.5, max=4.0, description="Similarity decay exponent")
    verbose: bool = config_field(False, description="Enable verbose logging")


class WindowSimilarity(TypedCallbackMixin[tuple[dict[int, Similarity], dict[int, LeaderScore]]]):
    """Computes pairwise window similarities in a background thread.

    Processes FeatureWindow data from active tracklets and computes similarity metrics
    for all pairs based on temporal patterns. Results are published via callbacks.

    Unlike FrameSimilarity which compares single frames, WindowSimilarity compares
    temporal sequences (windows) using time-series similarity algorithms like DTW,
    cross-correlation, or other sequence matching methods.

    Returns:
        Tuple of (similarity_dict, leader_dict) where:
        - similarity_dict: Maps track_id -> Similarity (magnitude of synchrony)
        - leader_dict: Maps track_id -> LeaderScore (temporal offset/who leads)
    """

    def __init__(self, config: WindowSimilarityConfig | None = None) -> None:
        super().__init__()

        self._config = config if config is not None else WindowSimilarityConfig()

        # Configure Similarity and LeaderScore modules with max_poses
        configure_similarity(self._config.max_poses)
        configure_leader_score(self._config.max_poses)

        self._correlation_thread = threading.Thread(target=self.run, daemon=True)
        self._stop_event = threading.Event()

        # INPUTS - Just store the latest data with a lock
        self._input_lock = threading.Lock()
        self._input_windows: WindowDict = {}
        self._update_event = threading.Event()

        # OUTPUT
        self._output_lock = threading.Lock()
        self._output_data: Optional[tuple[dict[int, 'Similarity'], dict[int, 'LeaderScore']]] = None

        self.Timer: PerformanceTimer = PerformanceTimer(name="similarity  ", sample_count=200, report_interval=100, color="red", omit_init=0)

        self._hot_reloader = HotReloadMethods(self.__class__, True, True)


    def start(self) -> None:
        """Start the similarity processing thread."""
        self._correlation_thread.start()

    def stop(self) -> None:
        """Stop the similarity processing thread and clear callbacks."""
        self._stop_event.set()
        self._update_event.set()  # Wake the thread so it can see stop_event

    def submit(self, windows: WindowDict) -> None:
        """Update input windows and trigger similarity processing.

        Args:
            windows: Dictionary mapping track IDs to FeatureWindow objects
        """
        with self._input_lock:
            self._input_windows = windows
            self._update_event.set()

    def run(self) -> None:
        """Main similarity processing loop (runs in background thread)."""
        while not self._stop_event.is_set():
            self._update_event.wait()
            if self._stop_event.is_set():
                break
            self._update_event.clear()

            try:
                # get input windows
                with self._input_lock:
                    windows: WindowDict = self._input_windows

                # Process similarity and leader scores
                start_time: float = time.perf_counter()
                result: tuple[dict[int, Similarity], dict[int, LeaderScore]] = self._process(windows)
                elapsed_time: float = (time.perf_counter() - start_time) * 1000.0
                self.Timer.add_time(elapsed_time, self._config.verbose)

                # Store and notify
                with self._output_lock:
                    self._output_data = result
                self._notify_callbacks(result)

            except Exception as e:
                print(f"WindowSimilarity: Processing error: {e}")
                traceback.print_exc()

    def _process(self, windows: WindowDict) -> tuple[dict[int,Similarity], dict[int, LeaderScore]]:
        """Process windows and return both Similarity and LeaderScore dicts."""

        if len(windows) < 2:
            return {}, {}

        # Step 1: Stack windows
        track_ids, values = self._stack_windows(windows)

        # Configure aggregator with joint count
        _, _, F = values.shape
        _JointAggregator.configure(F)

        # Step 2: Compute similarity tensor
        best_sim, confidence_scores, leader_scores = self._compute_similarity_tensor(values)
        # Step 3: Build per-pose dicts
        similarity_dict = self._build_similarity_dict(track_ids, best_sim, confidence_scores)
        leader_dict = self._build_leader_dict(track_ids, leader_scores)

        # Debug: Print per-pose similarities and leader scores
        if self._config.verbose:
            results = []
            for track_id in track_ids:
                sim_feature = similarity_dict[track_id]
                leader_feature = leader_dict[track_id]

                # Show similarities to other poses
                for other_id in range(self._config.max_poses):
                    if not np.isnan(sim_feature.values[other_id]):
                        sim_val = sim_feature.values[other_id]
                        conf_val = sim_feature.scores[other_id]
                        lead_val = leader_feature.values[other_id]
                        results.append(f"({track_id},{other_id}):sim={sim_val:.3f},conf={conf_val:.3f},lead={lead_val:+.2f}")

            print(f"Pairs: {' | '.join(results)}")

        return similarity_dict, leader_dict

    def _stack_windows(self, windows: WindowDict) -> tuple[list[int], np.ndarray]:
        """Stack all windows into (N, T, F) tensor.

        Returns:
            track_ids: List of track IDs
            values: Stacked window values (N, T, F)
        """
        window_length = self._config.window_length

        track_ids: list[int] = []
        slices: list[np.ndarray] = []

        for track_id, window in windows.items():
            track_ids.append(track_id)

            # Get last window_length samples (windows are always full-sized now)
            window_data = window.values[-window_length:]
            slices.append(window_data)

        # Stack: (N, T, F)
        values = np.stack(slices, axis=0)
        return track_ids, values

    def _compute_similarity_tensor(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pairwise similarity tensor using vectorized operations.

        Args:
            values: Stacked window values (N, T, F)

        Returns:
            best_sim: Best similarity per joint (N, N, F)
            confidence_scores: Confidence scores (N, N, F)
            leader_scores: Leader score [-1, 1] (N, N) - negative=first leads, positive=second leads, 0=sync
        """
        exponent = self._config.exponent
        window_length = self._config.window_length
        N, T, F = values.shape

        # Broadcast to compute all pairwise angular differences
        # values[:, None, :, None, :] -> (N, 1, T, 1, F)
        # values[None, :, None, :, :] -> (1, N, 1, T, F)
        # Result: (N, N, T, T, F) - all person pairs, all time pairs, all features
        raw_diff = values[:, None, :, None, :] - values[None, :, None, :, :]

        # Wrap angular difference to [-π, π] and compute similarity
        angular_diff = np.mod(raw_diff + np.pi, 2 * np.pi) - np.pi
        similarity = np.power(1.0 - np.abs(angular_diff) / np.pi, exponent)
        # NaN propagates automatically through all operations

        # Compute whole-body similarity per time pair (mean across features)
        # Weight by confidence (proportion of valid joints)
        # (N, N, T, T, F) -> (N, N, T, T)

        # Count valid joints per time pair
        valid_mask = ~np.isnan(similarity)  # (N, N, T, T, F)
        joint_count = np.sum(valid_mask, axis=4)  # (N, N, T, T)

        # Penalize by proportion of missing joints
        confidence_penalty = joint_count / F  # [0, 1]

        # Suppress warning for all-NaN slices (handled by nanmean returning NaN)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            whole_body_sim = np.nanmean(similarity, axis=4) * confidence_penalty

        # Find best (t_a, t_b) alignment per pair
        # Reshape to (N, N, T*T) for argmax, then convert back to (t_a, t_b)
        flat_indices = np.nanargmax(whole_body_sim.reshape(N, N, -1), axis=2)  # (N, N)
        t_a_indices = flat_indices // T  # Row index
        t_b_indices = flat_indices % T   # Column index

        # Compute leader scores: (t_b - t_a) / window_length → [-1, 1]
        # Negative = first person leads, Positive = second person leads, 0 = synchronized
        leader_scores = ((t_b_indices - t_a_indices) / window_length).astype(np.float32)  # (N, N)

        # Best per joint via nanmax over time axes
        # (N, N, T, T, F) -> (N, N, F)
        best_sim = np.nanmax(similarity, axis=(2, 3))
        best_sim = np.nan_to_num(best_sim, nan=0.0).astype(np.float32)

        # Scores from valid comparison proportion
        confidence_scores = 1.0 - np.mean(np.isnan(similarity), axis=(2, 3))
        confidence_scores = confidence_scores.astype(np.float32)

        return best_sim, confidence_scores, leader_scores

    def _build_similarity_dict(
        self,
        track_ids: list[int],
        best_sim: np.ndarray,
        confidence_scores: np.ndarray
    ) -> dict[int, 'Similarity']:
        """Build per-pose Similarity dict from similarity tensors.

        Args:
            track_ids: List of track IDs (N,)
            best_sim: Best similarity per joint (N, N, F)
            confidence_scores: Confidence scores (N, N, F)

        Returns:
            Dict mapping track_id -> Similarity object
        """
        from modules.pose.features.Similarity import Similarity

        N = len(track_ids)
        max_poses = self._config.max_poses
        result: dict[int, Similarity] = {}

        # For each pose i, build its Similarity array
        for i in range(N):
            values = np.full(max_poses, np.nan, dtype=np.float32)
            scores = np.zeros(max_poses, dtype=np.float32)

            # For each other pose j
            for j in range(N):
                if i == j:
                    continue  # Skip self-comparison

                # Use helper to aggregate per-joint similarities
                temp_feature = _JointAggregator(
                    values=best_sim[i, j],
                    scores=confidence_scores[i, j]
                )

                # Aggregate per-joint similarities into single scalar
                aggregated_sim = temp_feature.aggregate(
                    method=self._config.method,
                    min_confidence=0.0,
                    exponent=self._config.exponent
                )

                # Store at absolute pose ID index
                pose_id_j = track_ids[j]
                if not np.isnan(aggregated_sim):
                    values[pose_id_j] = aggregated_sim

                    # Compute mean confidence from valid joints
                    valid_scores = temp_feature.scores[temp_feature.valid_mask]
                    if len(valid_scores) > 0:
                        scores[pose_id_j] = np.mean(valid_scores)

            # Create Similarity object for this pose
            result[track_ids[i]] = Similarity(values, scores)

        return result

    def _build_leader_dict(
        self,
        track_ids: list[int],
        leader_scores: np.ndarray
    ) -> dict[int, 'LeaderScore']:
        """Build per-pose LeaderScore dict from leader score tensor.

        Args:
            track_ids: List of track IDs (N,)
            leader_scores: Temporal offset scores (N, N) in range [-1, 1]

        Returns:
            Dict mapping track_id -> LeaderScore object
        """
        from modules.pose.features.LeaderScore import LeaderScore

        N = len(track_ids)
        max_poses = self._config.max_poses
        result: dict[int, LeaderScore] = {}

        # For each pose i, build its LeaderScore array
        for i in range(N):
            values = np.zeros(max_poses, dtype=np.float32)
            scores = np.zeros(max_poses, dtype=np.float32)

            # For each other pose j
            for j in range(N):
                if i == j:
                    continue  # Skip self-comparison

                pose_id_j = track_ids[j]
                values[pose_id_j] = leader_scores[i, j]
                scores[pose_id_j] = 1.0  # Leader scores always valid when poses are compared

            # Create LeaderScore object for this pose
            result[track_ids[i]] = LeaderScore(values, scores)

        return result