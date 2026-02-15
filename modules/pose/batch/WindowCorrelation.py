# Standard library imports
from dataclasses import dataclass
from enum import IntEnum
import threading
import time
import traceback
from typing import Optional, cast
import warnings

# Third-party imports
import numpy as np
from numpy.fft import rfft, irfft

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
    """Helper class for aggregating per-joint correlations into scalars."""
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
class WindowCorrelationConfig(ConfigBase):
    """Configuration for WindowCorrelation."""
    max_poses: int = config_field(3, min=1, max=10, description="Maximum number of tracked poses")
    window_length: int = config_field(30, min=4, max=300, description="Number of frames for correlation")
    method: AggregationMethod = AggregationMethod.HARMONIC_MEAN
    max_lag: int = config_field(15, min=1, max=150, description="Maximum lag to search (±frames)")
    min_variance: float = config_field(0.01, min=0.001, max=0.5, description="Minimum variance for valid correlation")
    use_motion_weighting: bool = config_field(True, description="Weight correlation by motion magnitude")
    remap_low: float = config_field(0.0, min=-1.0, max=1.0, description="Correlation that maps to 0")
    remap_high: float = config_field(1.0, min=-1.0, max=1.0, description="Correlation that maps to 1")
    enabled: bool = config_field(False, description="Enable correlation computation")
    verbose: bool = config_field(False, description="Enable verbose logging")


class WindowCorrelation(TypedCallbackMixin[tuple[dict[int, Similarity], dict[int, LeaderScore]]]):
    """Computes pairwise window correlations in a background thread.

    Uses normalized cross-correlation per joint over full windows to detect
    synchrony and temporal offset between poses. More robust to scale differences
    than similarity-based approaches.

    Returns:
        Tuple of (correlation_dict, leader_dict) where:
        - correlation_dict: Maps track_id -> Similarity (peak correlation magnitude)
        - leader_dict: Maps track_id -> LeaderScore (lag at peak, normalized)
    """

    def __init__(self, config: WindowCorrelationConfig | None = None) -> None:
        super().__init__()

        self._config = config if config is not None else WindowCorrelationConfig()

        # Configure Similarity and LeaderScore modules with max_poses
        configure_similarity(self._config.max_poses)
        configure_leader_score(self._config.max_poses)

        self._correlation_thread = threading.Thread(target=self.run, daemon=True)
        self._stop_event = threading.Event()

        # INPUTS
        self._input_lock = threading.Lock()
        self._input_windows: WindowDict = {}
        self._input_motion_windows: WindowDict = {}
        self._update_event = threading.Event()

        # OUTPUT
        self._output_lock = threading.Lock()
        self._output_data: Optional[tuple[dict[int, 'Similarity'], dict[int, 'LeaderScore']]] = None

        self.Timer: PerformanceTimer = PerformanceTimer(name="correlation ", sample_count=200, report_interval=100, color="cyan", omit_init=0)

        self._hot_reloader = HotReloadMethods(self.__class__, True, True)

    def start(self) -> None:
        """Start the correlation processing thread."""
        self._correlation_thread.start()

    def stop(self) -> None:
        """Stop the correlation processing thread."""
        self._stop_event.set()
        self._update_event.set()

    def submit_all(self, all_windows: AllWindowDict) -> None:
        """Update all window types and trigger correlation processing.

        Args:
            all_windows: Combined dict {track_id: {FrameField: FeatureWindow}}
        """
        angle_windows: WindowDict = {}
        motion_windows: WindowDict = {}

        for track_id, field_windows in all_windows.items():
            if FrameField.angles in field_windows:
                angle_windows[track_id] = field_windows[FrameField.angles]
            if FrameField.angle_motion in field_windows:
                motion_windows[track_id] = field_windows[FrameField.angle_motion]

        with self._input_lock:
            self._input_windows = angle_windows
            self._input_motion_windows = motion_windows
            self._update_event.set()

    def run(self) -> None:
        """Main correlation processing loop (runs in background thread)."""
        while not self._stop_event.is_set():
            self._update_event.wait()
            if self._stop_event.is_set():
                break
            self._update_event.clear()

            try:
                with self._input_lock:
                    windows: WindowDict = self._input_windows
                    motion_windows: WindowDict = self._input_motion_windows

                start_time: float = time.perf_counter()
                result = self._process(windows, motion_windows)
                elapsed_time: float = (time.perf_counter() - start_time) * 1000.0
                self.Timer.add_time(elapsed_time, self._config.verbose)

                with self._output_lock:
                    self._output_data = result
                self._notify_callbacks(result)

            except Exception as e:
                print(f"WindowCorrelation: Processing error: {e}")
                traceback.print_exc()

    def _process(
        self,
        windows: WindowDict,
        motion_windows: WindowDict | None = None
    ) -> tuple[dict[int, Similarity], dict[int, LeaderScore]]:
        """Process windows and return correlation and leader score dicts."""

        if len(windows) < 2:
            return {}, {}

        track_ids = list(windows.keys())
        window_length = self._config.window_length

        # Stack angle windows (N, T, F)
        values = np.stack([w.values[-window_length:] for w in windows.values()], axis=0)
        N, T, F = values.shape

        _JointAggregator.configure(F)

        # Stack motion windows if enabled (N, T)
        motion_values: np.ndarray | None = None
        if self._config.use_motion_weighting and motion_windows:
            motion_values = np.stack([w.values[-window_length:, 0] for w in motion_windows.values()], axis=0)

        # Compute correlation tensor
        corr_matrix, lag_matrix, confidence_matrix = self._compute_correlation_tensor(values, motion_values)

        # Build output dicts
        similarity_dict = self._build_similarity_dict(track_ids, corr_matrix, confidence_matrix)
        leader_dict = self._build_leader_dict(track_ids, lag_matrix)

        if self._config.verbose:
            results = []
            for track_id in track_ids:
                sim_feature = similarity_dict[track_id]
                leader_feature = leader_dict[track_id]
                for other_id in range(self._config.max_poses):
                    if not np.isnan(sim_feature.values[other_id]):
                        sim_val = sim_feature.values[other_id]
                        lead_val = leader_feature.values[other_id]
                        results.append(f"({track_id},{other_id}):corr={sim_val:.3f},lag={lead_val:+.2f}")
            print(f"Pairs: {' | '.join(results)}")

        return similarity_dict, leader_dict

    def _compute_correlation_tensor(
        self,
        values: np.ndarray,
        motion_values: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pairwise correlation using vectorized FFT-based cross-correlation.

        Args:
            values: Stacked angle window values (N, T, F)
            motion_values: Stacked motion values (N, T) for weighting, or None

        Returns:
            corr_matrix: Peak correlation per joint (N, N, F) in [0, 1]
            lag_matrix: Lag at peak (N, N) normalized to [-1, 1]
            confidence_matrix: Confidence scores (N, N, F)
        """
        N, T, F = values.shape
        max_lag = min(self._config.max_lag, T - 1)
        min_var = self._config.min_variance
        fft_size = 2 * T

        # Compute variance per joint (N, F)
        variance = np.nanvar(values, axis=1)  # (N, F)

        # Mean-center the values
        means = np.nanmean(values, axis=1, keepdims=True)  # (N, 1, F)
        centered = values - means  # (N, T, F)

        # Transpose for FFT: (N, F, T)
        centered_t = np.transpose(centered, (0, 2, 1))

        # FFT all signals at once: (N, F, fft_size//2+1)
        all_fft = rfft(centered_t, n=fft_size, axis=2)

        # Cross-correlation for all pairs via broadcasting
        # a_fft[i] * conj(b_fft[j]) for all i,j
        # (N, 1, F, fft) * (1, N, F, fft) → (N, N, F, fft)
        cross_fft = all_fft[:, None, :, :] * np.conj(all_fft[None, :, :, :])

        # IRFFT back to time domain: (N, N, F, fft_size)
        xcorr_full = irfft(cross_fft, n=fft_size, axis=3)

        # Normalization factor: sqrt(var_i * var_j) * T per joint
        # (N, 1, F) * (1, N, F) → (N, N, F)
        norm_factor = np.sqrt(variance[:, None, :] * variance[None, :, :]) * T
        norm_factor = np.maximum(norm_factor, 1e-10)  # avoid div by zero

        # Normalize to Pearson correlation
        xcorr_full = xcorr_full / norm_factor[:, :, :, None]

        # Extract search window: lags from -max_lag to +max_lag
        # xcorr[..., 0] = lag 0, xcorr[..., 1] = lag 1, xcorr[..., -1] = lag -1
        positive_lags = xcorr_full[:, :, :, :max_lag + 1]  # lags 0 to max_lag
        negative_lags = xcorr_full[:, :, :, -max_lag:]     # lags -max_lag to -1
        search_window = np.concatenate([negative_lags, positive_lags], axis=3)  # (N, N, F, 2*max_lag+1)

        # Find peak correlation and lag per joint
        peak_idx = np.argmax(search_window, axis=3)  # (N, N, F)
        peak_corr = np.take_along_axis(search_window, peak_idx[:, :, :, None], axis=3).squeeze(3)  # (N, N, F)
        lag_per_joint = peak_idx - max_lag  # signed lag (N, N, F)

        # Confidence: 1.0 for valid variance, 0.5 for low variance
        low_var_mask = (variance[:, None, :] < min_var) | (variance[None, :, :] < min_var)  # (N, N, F)

        # Fallback for low variance: use mean angle difference
        mean_angles = np.nanmean(values, axis=1)  # (N, F)
        mean_diff = mean_angles[:, None, :] - mean_angles[None, :, :]  # (N, N, F)
        mean_diff = np.mod(mean_diff + np.pi, 2 * np.pi) - np.pi
        fallback_sim = np.exp(-np.square(mean_diff / 0.8))

        # Apply fallback where variance is low
        peak_corr_remapped = (peak_corr + 1) / 2  # remap [-1, 1] to [0, 1]
        corr_matrix = np.where(low_var_mask, fallback_sim, peak_corr_remapped).astype(np.float32)

        confidence_matrix = np.where(low_var_mask, 0.5, 1.0).astype(np.float32)

        # Handle NaN in input
        nan_mask = np.any(np.isnan(values), axis=1)  # (N, F)
        nan_pair_mask = nan_mask[:, None, :] | nan_mask[None, :, :]  # (N, N, F)
        corr_matrix = np.where(nan_pair_mask, np.nan, corr_matrix)
        confidence_matrix = np.where(nan_pair_mask, 0.0, confidence_matrix)

        # Average lag across joints weighted by |correlation|
        weights = np.abs(peak_corr)  # (N, N, F)
        weights = np.where(nan_pair_mask | low_var_mask, 0.0, weights)
        weight_sum = np.sum(weights, axis=2, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-10)
        avg_lag = np.sum(lag_per_joint * weights, axis=2) / weight_sum.squeeze(2)  # (N, N)

        # Normalize lag to [-1, 1]
        lag_matrix = (avg_lag / max_lag).astype(np.float32) if max_lag > 0 else np.zeros((N, N), dtype=np.float32)

        # Motion weighting
        if motion_values is not None:
            mean_motion = np.nanmean(motion_values, axis=1)  # (N,)
            motion_weight = mean_motion[:, None] * mean_motion[None, :]  # (N, N)
            corr_matrix = corr_matrix * motion_weight[:, :, None]

        return corr_matrix, lag_matrix, confidence_matrix

    def _build_similarity_dict(
        self,
        track_ids: list[int],
        corr_matrix: np.ndarray,
        confidence_matrix: np.ndarray
    ) -> dict[int, Similarity]:
        """Build per-pose Similarity dict from correlation matrix."""
        from modules.pose.features.Similarity import Similarity

        N = len(track_ids)
        max_poses = self._config.max_poses
        result: dict[int, Similarity] = {}

        for i in range(N):
            values = np.full(max_poses, np.nan, dtype=np.float32)
            scores = np.zeros(max_poses, dtype=np.float32)

            for j in range(N):
                if i == j:
                    continue

                temp_feature = _JointAggregator(
                    values=corr_matrix[i, j],
                    scores=confidence_matrix[i, j]
                )

                aggregated = temp_feature.aggregate(
                    method=self._config.method,
                    min_confidence=0.0
                )

                # Apply remap
                low, high = self._config.remap_low, self._config.remap_high
                if high > low and not np.isnan(aggregated):
                    aggregated = (aggregated - (low + 1) / 2) / ((high - low) / 2)
                    aggregated = float(np.clip(aggregated, 0.0, 1.0))

                pose_id_j = track_ids[j]
                if not np.isnan(aggregated):
                    values[pose_id_j] = aggregated
                    valid_scores = temp_feature.scores[temp_feature.valid_mask]
                    if len(valid_scores) > 0:
                        scores[pose_id_j] = np.mean(valid_scores)

            result[track_ids[i]] = Similarity(values, scores)

        return result

    def _build_leader_dict(
        self,
        track_ids: list[int],
        lag_matrix: np.ndarray
    ) -> dict[int, LeaderScore]:
        """Build per-pose LeaderScore dict from lag matrix."""
        from modules.pose.features.LeaderScore import LeaderScore

        N = len(track_ids)
        max_poses = self._config.max_poses
        result: dict[int, LeaderScore] = {}

        for i in range(N):
            values = np.zeros(max_poses, dtype=np.float32)
            scores = np.zeros(max_poses, dtype=np.float32)

            for j in range(N):
                if i == j:
                    continue

                pose_id_j = track_ids[j]
                # Normalize lag to [0, 1] range: 0 = sync, 1 = j leads by max_lag
                values[pose_id_j] = (lag_matrix[i, j] + 1) / 2
                scores[pose_id_j] = 1.0

            result[track_ids[i]] = LeaderScore(values, scores)

        return result
