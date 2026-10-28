# Standard library imports
from enum import IntEnum
import math
import threading
import time
from typing import Callable, Optional, TYPE_CHECKING, cast
import warnings
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Pose imports
from modules.settings import BaseSettings, Field, Group
from ..frame import FeatureWindow, FeatureWindowDict, FrameWindowDict
from ..features import Angles, AngleMotion, AngleVelocity
from ..features import AggregationMethod, NormalizedScalarFeature
from ..features import Similarity, LeaderScore
from .joint_select import JointSelectSettings, joint_mask
from modules.utils import PerformanceTimer, HotReloadMethods

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Output of WindowSimilarity / WindowCorrelation / PostureSimilarity."""
    similarity: dict[int, Similarity]
    leader_score: dict[int, LeaderScore]


class _JointAggregator(NormalizedScalarFeature):
    """Per-joint similarities as a feature, so NormalizedScalarFeature's aggregation methods (mean, harmonic
    mean, …) apply to them. Configured once with the joint count (class-level, idempotent)."""
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


def joint_similarity(diff: np.ndarray, tolerance: float) -> np.ndarray:
    """Per joint ``exp(-(Δ / tolerance)²)`` on the angle differences ``diff`` (radians, any shape), wrapped to
    the shortest way round; NaN where a joint is missing stays NaN. ``tolerance`` in radians: 1/e at one."""
    wrapped = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    return np.exp(-np.square(wrapped / tolerance))


def aggregate_joints(joint_sims: np.ndarray, method: AggregationMethod,
                     remap_low: float, remap_high: float, mask: np.ndarray | None = None) -> tuple[float, float]:
    """One pair's ``(value, coverage)`` from its per-joint similarities (F,), NaN for joints not compared: the
    aggregate over the joints present (``method``), remapped ``[remap_low, remap_high] → [0, 1]`` when the
    range is positive, times the fraction of joints compared. A joint ``mask`` (bool (F,)) leaves out is
    neither compared nor counted, so the coverage is over the selected joints. ``(NaN, 0.0)`` when no joint
    was compared."""
    joint_sims = np.asarray(joint_sims, dtype=np.float32)
    if mask is not None:
        joint_sims = np.where(mask, joint_sims, np.nan)
    valid = ~np.isnan(joint_sims)
    included = joint_sims.size if mask is None else int(np.sum(mask))
    if included == 0 or not valid.any():
        return math.nan, 0.0
    coverage = float(valid.sum() / included)
    _JointAggregator.configure(int(joint_sims.size))
    feature = _JointAggregator(values=joint_sims.copy(), scores=valid.astype(np.float32))
    value = feature.aggregate(method=method, min_confidence=0.0)
    if math.isnan(value):
        return math.nan, 0.0
    if remap_high > remap_low:
        value = float(np.clip((value - remap_low) / (remap_high - remap_low), 0.0, 1.0))
    return value * coverage, coverage


class WindowSimilaritySettings(BaseSettings):
    """Configuration for WindowSimilarity."""
    max_poses:              Field[int]                = Field(3, min=1, max=10, access=Field.INIT, description="Maximum number of tracked poses")
    window_length:          Field[int]                = Field(30, min=1, max=300, description="Number of frames to compare")
    method:                 Field[AggregationMethod]  = Field(AggregationMethod.HARMONIC_MEAN, description="Aggregation method")
    use_angle_similarity:   Field[bool]               = Field(True, description="Use angle similarity")
    angle_tolerance:        Field[float]              = Field(45.0, min=1.0, max=120.0, step=1.0, description="Joint angles this far apart read e⁻¹ (°); the one knob for how alike, smaller is stricter")
    joints:                 Group[JointSelectSettings] = Group(JointSelectSettings)
    use_velocity_similarity: Field[bool]              = Field(True, description="Multiply by velocity similarity")
    velocity_tolerance:     Field[float]              = Field(30.0, min=1.0, max=120.0, step=1.0, description="Joint velocities this far apart still count as similar (°/s); smaller is stricter")
    use_motion_weighting:   Field[bool]               = Field(True, description="Weight similarity by motion")
    use_time_penalty:       Field[bool]               = Field(True, description="Penalize older frames in window")
    time_decay_exp:         Field[float]              = Field(1.0, min=0.1, max=4.0, description="Time decay exponent")
    remap_low:              Field[float]              = Field(0.05, min=0.0, max=1.0, description="Raw similarity that maps to 0; 0 with remap_high 1 leaves the raw similarity")
    remap_high:             Field[float]              = Field(0.8, min=0.0, max=1.0, description="Raw similarity that maps to 1; 1 with remap_low 0 leaves the raw similarity")
    enabled:                Field[bool]               = Field(True, description="Enable similarity computation")
    verbose:                Field[bool]               = Field(False, description="Enable verbose logging")


class WindowSimilarity:
    """Computes pairwise window similarities in a background thread.

    Processes FeatureWindow data from active tracklets and computes similarity metrics
    for all pairs based on temporal patterns. Results are published via callbacks.

    Unlike FrameSimilarity which compares single frames, WindowSimilarity compares
    temporal sequences (windows) using time-series similarity algorithms like DTW,
    cross-correlation, or other sequence matching methods.

    ``joints`` selects the joints compared and counted: an unchecked joint is dropped before the per-joint
    similarities are aggregated, and the coverage is the fraction of the selected joints both people have.
    """

    def __init__(self, config: WindowSimilaritySettings | None = None) -> None:

        self._config = config if config is not None else WindowSimilaritySettings()

        self._correlation_thread = threading.Thread(target=self.run, daemon=True)
        self._stop_event = threading.Event()

        # INPUTS - Just store the latest data with a lock
        self._input_lock = threading.Lock()
        self._input_windows: FeatureWindowDict = {}  # Angle windows (required)
        self._input_motion_windows: FeatureWindowDict = {}  # AngleMotion windows (optional)
        self._input_velocity_windows: FeatureWindowDict = {}  # Angle velocity windows (optional)
        self._update_event = threading.Event()

        # OUTPUT
        self._output_lock = threading.Lock()
        self._output_data: SimilarityResult | None = None
        self._callbacks: list[Callable[[SimilarityResult], None]] = []      # run in registration order

        self.Timer: PerformanceTimer = PerformanceTimer(name="similarity  ", sample_count=200, report_interval=100, color="red", omit_init=0)

        self._hot_reloader = HotReloadMethods(self.__class__, True, True)


    def start(self) -> None:
        """Start the similarity processing thread."""
        self._correlation_thread.start()

    def stop(self) -> None:
        """Stop the similarity processing thread and clear callbacks."""
        self._stop_event.set()
        self._update_event.set()  # Wake the thread so it can see stop_event

    def add_similarity_callback(self, callback: Callable[[SimilarityResult], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def submit(self, all_windows: FrameWindowDict) -> None:
        """Submit all window types for similarity processing.

        Extracts angle, motion, and velocity windows from the combined
        WindowTracker output format.

        Args:
             all_windows: Combined dict {type[BaseFeature]: {track_id: FeatureWindow}}
        """
        if not self._config.enabled:
            return

        # Extract per-field window dicts
        angle_windows: FeatureWindowDict = all_windows.get(Angles, {})
        motion_windows: FeatureWindowDict = all_windows.get(AngleMotion, {})
        velocity_windows: FeatureWindowDict = all_windows.get(AngleVelocity, {})

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
                    windows: FeatureWindowDict = self._input_windows
                    motion_windows: FeatureWindowDict = self._input_motion_windows
                    velocity_windows: FeatureWindowDict = self._input_velocity_windows

                # Process similarity and leader scores
                start_time: float = time.perf_counter()
                similarity_dict, leader_dict = self._process(
                    windows, motion_windows, velocity_windows
                )
                elapsed_time: float = (time.perf_counter() - start_time) * 1000.0
                self.Timer.add_time(elapsed_time, self._config.verbose)

                result = SimilarityResult(similarity_dict, leader_dict)
                with self._output_lock:
                    self._output_data = result
                for cb in self._callbacks:
                    cb(result)

            except Exception as e:
                logger.error(f"Processing error: {e}")
    def _process(
        self,
        windows: FeatureWindowDict,
        motion_windows: FeatureWindowDict | None = None,
        velocity_windows: FeatureWindowDict | None = None
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

        # Step 1: Stack angle windows (N, T, F); missing joints and unfilled slots are NaN
        values = np.stack([w.masked_values()[-window_length:] for w in windows.values()], axis=0)

        # Configure aggregator with joint count
        _, _, F = values.shape
        _JointAggregator.configure(F)

        # Step 1b: Stack motion windows if enabled (N, T); missing motion stays 0 (no motion), as NaN would
        # blank the whole body at that frame
        motion_values: np.ndarray | None = None
        if self._config.use_motion_weighting and motion_windows:
            motion_values = np.stack([w.values[-window_length:, 0] for w in motion_windows.values()], axis=0)

        # Step 1c: Stack velocity windows if enabled (N, T, F)
        velocity_values: np.ndarray | None = None
        if self._config.use_velocity_similarity and velocity_windows:
            velocity_values = np.stack([w.masked_values()[-window_length:] for w in velocity_windows.values()], axis=0)

        # Step 2: Compute similarity tensor over the selected joints
        mask = joint_mask(self._config.joints)
        best_sim, coverage, leader_scores = self._compute_similarity_tensor(values, motion_values, velocity_values, mask)
        # Step 3: Build per-pose dicts
        similarity_dict = self._build_similarity_dict(track_ids, best_sim, coverage, mask)
        leader_dict = self._build_leader_dict(track_ids, leader_scores, coverage)

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

            logger.info(f"Pairs: {' | '.join(results)}")

        return similarity_dict, leader_dict

    def _compute_similarity_tensor(
        self,
        values: np.ndarray,
        motion_values: np.ndarray | None = None,
        velocity_values: np.ndarray | None = None,
        mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pairwise similarity tensor using current×all comparison.

        Compares each person's current frame against all frames in others' windows.
        Uses Gaussian similarity for both angles and velocities.

        Missing data is NaN. A joint missing in either person's angles drops out of that comparison; missing
        velocity leaves the angle similarity unscaled, so joint coverage is decided by angles alone. A joint
        ``mask`` leaves out is neither compared nor counted: coverage is over the selected joints.

        Args:
            values: Stacked angle window values (N, T, F), NaN where missing
            motion_values: Stacked motion values (N, T) for weighting, or None
            velocity_values: Stacked velocity values (N, T, F) for similarity, NaN where missing, or None
            mask: The joints selected, bool (F,); None selects all

        Returns:
            best_sim: Similarity per joint at the best-matching frame (N, N, F), NaN for joints not compared
            coverage: Fraction of selected joints compared at the best-matching frame (N, N), 0 when nothing overlapped
            leader_scores: Leader score [0, 1] (N, N) - 0=sync, 1=j leads by full window
        """
        angle_tolerance = np.radians(self._config.angle_tolerance)          # the settings are degrees, the angles radians
        velocity_tolerance = np.radians(self._config.velocity_tolerance)
        N, T, F = values.shape
        included = F if mask is None else int(np.sum(mask))

        # Current frame of each person: (N, F)
        current = values[:, -1, :]

        # Broadcast: current[:, None, None, :] -> (N, 1, 1, F)
        #            values[None, :, :, :]     -> (1, N, T, F)
        # Result: (N, N, T, F) - i's current vs j's full window
        raw_diff = current[:, None, None, :] - values[None, :, :, :]

        # Compute angle similarity if enabled: the shared posture kernel (posture_similarity.py)
        if self._config.use_angle_similarity:
            similarity = joint_similarity(raw_diff, angle_tolerance)
        else:
            # Start with ones (neutral), NaN where input is NaN
            similarity = np.where(np.isnan(raw_diff), np.nan, 1.0)

        # Apply velocity similarity if enabled (per-joint)
        if velocity_values is not None:
            vel_current = velocity_values[:, -1, :]  # (N, F)
            vel_diff = vel_current[:, None, None, :] - velocity_values[None, :, :, :]  # (N, N, T, F)
            vel_sim = np.exp(-np.square(vel_diff / velocity_tolerance))
            vel_sim = np.where(np.isnan(vel_sim), 1.0, vel_sim)  # missing velocity is neutral
            similarity = similarity * vel_sim  # element-wise per joint

        # Drop the joints not selected: they are neither compared nor counted
        if mask is not None:
            similarity = np.where(mask[None, None, None, :], similarity, np.nan)

        # Compute whole-body similarity per time step (mean across features)
        # (N, N, T, F) -> (N, N, T)

        # Count valid joints per time step
        valid_mask = ~np.isnan(similarity)  # (N, N, T, F)
        joint_count = np.sum(valid_mask, axis=3)  # (N, N, T)

        # Penalize by proportion of the selected joints missing
        confidence_penalty = joint_count / included if included > 0 else np.zeros_like(joint_count, dtype=np.float32)  # [0, 1]

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

        # Find best match time index for each pair; a pair with no overlapping data at any t has no best
        # match (nanargmax would raise), it gets t=0 and zero coverage below
        best_t = np.argmax(np.where(np.isnan(whole_body_sim), -np.inf, whole_body_sim), axis=2)  # (N, N)

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
        best_sim = similarity[i_idx, j_idx, best_t, :].astype(np.float32)  # (N, N, F)

        # Fraction of the selected joints compared at best_t
        compared = np.sum(~np.isnan(best_sim), axis=2)
        coverage = (compared / included if included > 0 else np.zeros_like(compared)).astype(np.float32)  # (N, N)

        return best_sim, coverage, leader_scores

    def _build_similarity_dict(
        self,
        track_ids: list[int],
        best_sim: np.ndarray,
        coverage: np.ndarray,
        mask: np.ndarray | None = None
    ) -> dict[int, 'Similarity']:
        """Build per-pose Similarity dict from similarity tensors.

        Each pair's value aggregates the selected joints both people have, remapped, then scaled by joint
        coverage so a mostly-occluded pair can't read as fully in sync. The score is the coverage; a pair with
        no joint in common is NaN with score 0.

        Args:
            track_ids: List of track IDs (N,)
            best_sim: Similarity per joint (N, N, F), NaN for joints not compared
            coverage: Fraction of the selected joints compared per pair (N, N)
            mask: The joints selected, bool (F,); None selects all

        Returns:
            Dict mapping track_id -> Similarity object
        """

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

                # The shared posture kernel's aggregation (posture_similarity.py): the joints both have,
                # remapped, times the coverage; NaN joints are skipped
                value, _ = aggregate_joints(best_sim[i, j], self._config.method,
                                            self._config.remap_low, self._config.remap_high, mask)

                # Store at absolute pose ID index; the score is the fraction of joints compared
                pose_id_j = track_ids[j]
                if not np.isnan(value):
                    values[pose_id_j] = value
                    scores[pose_id_j] = coverage[i, j]

            # Create Similarity object for this pose
            result[track_ids[i]] = Similarity(values, scores)

        return result

    def _build_leader_dict(
        self,
        track_ids: list[int],
        leader_scores: np.ndarray,
        coverage: np.ndarray
    ) -> dict[int, 'LeaderScore']:
        """Build per-pose LeaderScore dict from leader score tensor.

        A pair with no joint in common has no meaningful best match: value 0 with score 0.

        Args:
            track_ids: List of track IDs (N,)
            leader_scores: Temporal offset scores (N, N) in range [0, 1]
            coverage: Fraction of joints compared per pair (N, N)

        Returns:
            Dict mapping track_id -> LeaderScore object
        """

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
                if coverage[i, j] > 0.0:
                    values[pose_id_j] = leader_scores[i, j]
                    scores[pose_id_j] = 1.0

            # Create LeaderScore object for this pose
            result[track_ids[i]] = LeaderScore(values, scores)

        return result