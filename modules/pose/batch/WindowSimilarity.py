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
from modules.pose.Frame import FrameField
from modules.pose.nodes.windows.WindowNode import FeatureWindow
from modules.pose.features.base.NormalizedScalarFeature import AggregationMethod, NormalizedScalarFeature
from modules.pose.features.Similarity import configure_similarity, Similarity
from modules.pose.features.LeaderScore import configure_leader_score, LeaderScore
from modules.utils.PerformanceTimer import PerformanceTimer

from modules.utils.HotReloadMethods import HotReloadMethods

# Type alias for window dictionary (track_id -> FeatureWindow)
WindowDict = dict[int, FeatureWindow]

# Type alias for combined window dict from AllWindowTracker: {track_id: {FrameField: FeatureWindow}}
AllWindowDict = dict[int, dict[FrameField, FeatureWindow]]


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
    use_angle_similarity: bool = config_field(True, description="Use angle similarity")
    angle_scale: float = config_field(0.8, min=0.1, max=2.0, description="Angle similarity scale (rad, ~π/4 for steep)")
    use_velocity_similarity: bool = config_field(True, description="Multiply by velocity similarity")
    vel_scale: float = config_field(0.5, min=0.1, max=2.0, description="Velocity similarity scale (rad/frame)")
    use_motion_weighting: bool = config_field(True, description="Weight similarity by motion (motion_i × motion_j)")
    use_time_penalty: bool = config_field(True, description="Penalize matches to older frames in window")
    time_decay_exp: float = config_field(1.0, min=0.1, max=4.0, description="Time decay exponent (1.0=linear, >1=exponential)")
    remap_low: float = config_field(0.05, min=0.0, max=1.0, description="Raw similarity that maps to 0")
    remap_high: float = config_field(0.8, min=0.0, max=1.0, description="Raw similarity that maps to 1")
    enabled: bool = config_field(True, description="Enable similarity computation")
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
        self._input_windows: WindowDict = {}  # Angle windows (required)
        self._input_motion_windows: WindowDict = {}  # AngleMotion windows (optional)
        self._input_velocity_windows: WindowDict = {}  # Angle velocity windows (optional)
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
        """Update angle windows and trigger similarity processing.

        DEPRECATED: Use submit_all() for motion gate and velocity weighting.

        Args:
            windows: Dictionary mapping track IDs to angle FeatureWindow objects
        """
        with self._input_lock:
            self._input_windows = windows
            self._update_event.set()

    def submit_all(self, all_windows: AllWindowDict) -> None:
        """Update all window types and trigger similarity processing.

        Extracts angle, motion, and velocity windows from the combined
        AllWindowTracker output format.

        Args:
            all_windows: Combined dict {track_id: {FrameField: FeatureWindow}}
        """
        if not self._config.enabled:
            return

        # Extract per-field window dicts
        angle_windows: WindowDict = {}
        motion_windows: WindowDict = {}
        velocity_windows: WindowDict = {}

        for track_id, field_windows in all_windows.items():
            if FrameField.angles in field_windows:
                angle_windows[track_id] = field_windows[FrameField.angles]
            if FrameField.angle_motion in field_windows:
                motion_windows[track_id] = field_windows[FrameField.angle_motion]
            if FrameField.angle_vel in field_windows:
                velocity_windows[track_id] = field_windows[FrameField.angle_vel]

        with self._input_lock:
            self._input_windows = angle_windows
            self._input_motion_windows = motion_windows
            self._input_velocity_windows = velocity_windows
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
                    motion_windows: WindowDict = self._input_motion_windows
                    velocity_windows: WindowDict = self._input_velocity_windows

                # Process similarity and leader scores
                start_time: float = time.perf_counter()
                result: tuple[dict[int, Similarity], dict[int, LeaderScore]] = self._process(
                    windows, motion_windows, velocity_windows
                )
                elapsed_time: float = (time.perf_counter() - start_time) * 1000.0
                self.Timer.add_time(elapsed_time, self._config.verbose)

                # Store and notify
                with self._output_lock:
                    self._output_data = result
                self._notify_callbacks(result)

            except Exception as e:
                print(f"WindowSimilarity: Processing error: {e}")
                traceback.print_exc()

    def _process(
        self,
        windows: WindowDict,
        motion_windows: WindowDict | None = None,
        velocity_windows: WindowDict | None = None
    ) -> tuple[dict[int,Similarity], dict[int, LeaderScore]]:
        """Process windows and return both Similarity and LeaderScore dicts.

        Args:
            windows: Angle windows (required)
            motion_windows: AngleMotion windows for weighting (optional)
            velocity_windows: Angle velocity windows for sign weighting (optional)
        """

        if len(windows) < 2:
            return {}, {}

        # Extract track IDs (same order as dict iteration)
        track_ids = list(windows.keys())
        window_length = self._config.window_length

        # Step 1: Stack angle windows (N, T, F)
        values = np.stack([w.values[-window_length:] for w in windows.values()], axis=0)

        # Configure aggregator with joint count
        _, _, F = values.shape
        _JointAggregator.configure(F)

        # Step 1b: Stack motion windows if enabled (N, T)
        motion_values: np.ndarray | None = None
        if self._config.use_motion_weighting and motion_windows:
            motion_values = np.stack([w.values[-window_length:, 0] for w in motion_windows.values()], axis=0)

        # Step 1c: Stack velocity windows if enabled (N, T, F)
        velocity_values: np.ndarray | None = None
        if self._config.use_velocity_similarity and velocity_windows:
            velocity_values = np.stack([w.values[-window_length:] for w in velocity_windows.values()], axis=0)

        # Step 2: Compute similarity tensor
        best_sim, confidence_scores, leader_scores = self._compute_similarity_tensor(values, motion_values, velocity_values)
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

    def _compute_similarity_tensor(
        self,
        values: np.ndarray,
        motion_values: np.ndarray | None = None,
        velocity_values: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pairwise similarity tensor using current×all comparison.

        Compares each person's current frame against all frames in others' windows.
        Uses Gaussian similarity for both angles and velocities.

        Args:
            values: Stacked angle window values (N, T, F)
            motion_values: Stacked motion values (N, T) for weighting, or None
            velocity_values: Stacked velocity values (N, T, F) for similarity, or None

        Returns:
            best_sim: Best similarity per joint (N, N, F)
            confidence_scores: Confidence scores (N, N, F)
            leader_scores: Leader score [0, 1] (N, N) - 0=sync, 1=j leads by full window
        """
        # print(self._config)

        angle_scale = self._config.angle_scale
        vel_scale = self._config.vel_scale
        N, T, F = values.shape

        # Current frame of each person: (N, F)
        current = values[:, -1, :]

        # Broadcast: current[:, None, None, :] -> (N, 1, 1, F)
        #            values[None, :, :, :]     -> (1, N, T, F)
        # Result: (N, N, T, F) - i's current vs j's full window
        raw_diff = current[:, None, None, :] - values[None, :, :, :]

        # Compute angle similarity if enabled
        if self._config.use_angle_similarity:
            angular_diff = np.mod(raw_diff + np.pi, 2 * np.pi) - np.pi
            similarity = np.exp(-np.square(angular_diff / angle_scale))
        else:
            # Start with ones (neutral), NaN where input is NaN
            similarity = np.where(np.isnan(raw_diff), np.nan, 1.0)

        # Apply velocity similarity if enabled (per-joint)
        if velocity_values is not None:
            vel_current = velocity_values[:, -1, :]  # (N, F)
            vel_diff = vel_current[:, None, None, :] - velocity_values[None, :, :, :]  # (N, N, T, F)
            vel_sim = np.exp(-np.square(vel_diff / vel_scale))
            similarity = similarity * vel_sim  # element-wise per joint

        # Compute whole-body similarity per time step (mean across features)
        # (N, N, T, F) -> (N, N, T)

        # Count valid joints per time step
        valid_mask = ~np.isnan(similarity)  # (N, N, T, F)
        joint_count = np.sum(valid_mask, axis=3)  # (N, N, T)

        # Penalize by proportion of missing joints
        confidence_penalty = joint_count / F  # [0, 1]

        # Suppress warning for all-NaN slices (handled by nanmean returning NaN)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            whole_body_sim = np.nanmean(similarity, axis=3) * confidence_penalty  # (N, N, T)

        # Apply time penalty: favor recent matches (t close to T-1)
        if self._config.use_time_penalty and T > 1:
            # t=0 oldest → t=T-1 newest (current)
            t_normalized = np.arange(T) / (T - 1)  # [0, 1]
            time_weight = np.power(t_normalized, self._config.time_decay_exp)  # (T,)
            whole_body_sim = whole_body_sim * time_weight  # broadcast (N, N, T)

        # Apply motion weighting: motion_i[current] × motion_j[t]
        # This affects BOTH leader selection AND final similarity score
        if motion_values is not None:
            # motion_values[:, -1] -> (N,) current motion
            # motion_values        -> (N, T) full window
            # Result: (N, N, T) -> broadcast to (N, N, T, 1) for per-joint
            motion_weight = motion_values[:, None, -1:] * motion_values[None, :, :]  # (N, N, T)
            whole_body_sim = whole_body_sim * motion_weight
            # Apply to per-joint similarity tensor too
            similarity = similarity * motion_weight[:, :, :, None]  # (N, N, T, F)

        # Find best match time index for each pair
        best_t = np.nanargmax(whole_body_sim, axis=2)  # (N, N)

        # Leader score: (T-1 - best_t) / (T-1) → 0=sync, 1=j leads by full window
        # When best_t = T-1 (matched current): leader = 0 (synchronized)
        # When best_t = 0 (matched oldest): leader = 1 (j leads)
        if T > 1:
            leader_scores = ((T - 1 - best_t) / (T - 1)).astype(np.float32)  # (N, N)
        else:
            leader_scores = np.zeros((N, N), dtype=np.float32)

        # Extract per-joint similarity AT best_t (not max over all time)
        # (N, N, T, F) -> (N, N, F) using advanced indexing
        i_idx = np.arange(N)[:, None]  # (N, 1)
        j_idx = np.arange(N)[None, :]  # (1, N)
        best_sim = similarity[i_idx, j_idx, best_t, :]  # (N, N, F)
        best_sim = np.nan_to_num(best_sim, nan=0.0).astype(np.float32)

        # Confidence scores at best_t
        confidence_scores = 1.0 - np.isnan(similarity[i_idx, j_idx, best_t, :]).astype(np.float32)  # (N, N, F)

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
                    min_confidence=0.0
                )

                # Remap [remap_low, remap_high] → [0, 1]
                low, high = self._config.remap_low, self._config.remap_high
                # print(low, high)
                if high > low and not np.isnan(aggregated_sim):
                    aggregated_sim = (aggregated_sim - low) / (high - low)
                    aggregated_sim = float(np.clip(aggregated_sim, 0.0, 1.0))

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