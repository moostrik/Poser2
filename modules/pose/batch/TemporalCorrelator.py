"""Motion synchrony analyzer via bounded-lag correlation.

Consumes BufferOutput from RollingFeatureBuffer and computes pairwise motion
synchrony using bounded-lag cross-correlation of joint angular velocities,
weighted by per-joint motion energy and aggregated via harmonic mean.

Outputs SimilarityBatch for unified downstream processing with static similarity.
"""

from dataclasses import dataclass
from itertools import combinations
import threading
import time

import numpy as np
import torch

from modules.ConfigBase import ConfigBase, config_field
from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.similarity.features.SimilarityFeature import SimilarityFeature
from modules.pose.similarity.features.SimilarityBatch import SimilarityBatch

from modules.utils.PerformanceTimer import PerformanceTimer
from modules.utils.HotReloadMethods import HotReloadMethods

# Type alias for input (from RollingFeatureBuffer)
BufferOutput = tuple[torch.Tensor, torch.Tensor]


@dataclass
class TemporalCorrelatorConfig(ConfigBase):
    """Configuration for motion synchrony computation.

    Bounded-lag correlation with motion energy weighting.
    """

    max_lag_frames: int = config_field(
        15, min=1, max=60,
        description="Maximum lag in frames for cross-correlation (e.g., 8 frames @ 23fps = 348ms)"
    )
    energy_low: float = config_field(
        0.1, min=0.0, max=10.0,
        description="Motion energy threshold below which joint is considered stationary (rad/s RMS)"
    )
    energy_high: float = config_field(
        0.5, min=0.1, max=20.0,
        description="Motion energy threshold for full contribution (rad/s RMS)"
    )
    ema_alpha: float = config_field(
        0.8, min=0.0, max=1.0,
        description="EMA smoothing factor (1.0 = no smoothing, 0.0 = infinite smoothing)"
    )
    eps: float = config_field(
        1e-6, min=1e-10, max=1e-3,
        description="Epsilon for numerical stability in harmonic mean"
    )


class TemporalCorrelator(TypedCallbackMixin[SimilarityBatch]):
    """GPU-based motion synchrony analyzer with async callback dispatch.

    Computes pairwise motion synchrony via bounded-lag correlation of joint
    angular velocities, weighted by per-joint motion energy.

    Pipeline:
    1. Compute motion energy: RMS of velocity over time window
    2. Compute support weights: Clamp energy to [0,1] range
    3. Vectorized bounded-lag correlation: All pairs via einsum
    4. Motion gating: Weight correlation by support weights
    5. EMA smoothing: Temporal stability
    6. Output as SimilarityBatch for unified downstream processing

    Architecture:
    - submit() receives BufferOutput, stores refs (CPU-only, ~1-5μs)
    - Background thread handles GPU computation and callbacks
    - Never blocks the pose processing thread

    Input: (num_tracks, window_size, feature_length) angular velocity buffer
    Output: SimilarityBatch containing SimilarityFeature per unique pair
        - values: per-joint synchrony in [0, 1]
        - scores: combined support weights (support_i * support_j)
    """

    def __init__(self, config: TemporalCorrelatorConfig) -> None:
        super().__init__()

        self._config = config
        self._max_lag = config.max_lag_frames
        self._energy_low = config.energy_low
        self._energy_high = config.energy_high
        self._ema_alpha = config.ema_alpha
        self._eps = config.eps

        # EMA state - GPU tensor, initialized on first frame
        self._ema_synchrony: torch.Tensor = torch.empty(0)
        self._ema_initialized: bool = False

        # Precomputed pair indices (set on first frame with known num_tracks)
        self._pair_indices: list[tuple[int, int]] = []
        self._pair_i: torch.Tensor = torch.empty(0, dtype=torch.long)  # GPU indices for gather
        self._pair_j: torch.Tensor = torch.empty(0, dtype=torch.long)

        # Pre-allocated buffers (initialized on first frame)
        self._x_normalized: torch.Tensor = torch.empty(0)  # (N, W, J) normalized velocities
        self._corr_max: torch.Tensor = torch.empty(0)      # (N, N, J) max correlation buffer
        self._corr_temp: torch.Tensor = torch.empty(0)     # (N, N, J) temp for einsum result

        # Staging references (CPU-side, written in submit)
        self._staging_values: torch.Tensor | None = None
        self._staging_mask: torch.Tensor | None = None

        # Async notification thread
        self._notify_event = threading.Event()
        self._shutdown_flag = False
        self._started = False
        self._notify_thread: threading.Thread | None = None

        # Track frame count
        self._frame_count = 0

        self._T1: PerformanceTimer = PerformanceTimer(name="prepare  ", sample_count=200, report_interval=100, color="red", omit_init=25)
        self._T2: PerformanceTimer = PerformanceTimer(name="correlate", sample_count=200, report_interval=100, color="green", omit_init=25)
        self._T3: PerformanceTimer = PerformanceTimer(name="finish   ", sample_count=200, report_interval=100, color="blue", omit_init=25)
        self._T4: PerformanceTimer = PerformanceTimer(name="download ", sample_count=200, report_interval=100, color="yellow", omit_init=25)


        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def start(self) -> None:
        """Start the async processing thread.

        Must be called before submit() to enable async processing.
        Safe to call multiple times (subsequent calls are no-op).
        """
        if self._started:
            return

        self._shutdown_flag = False
        self._notify_thread = threading.Thread(
            target=self._process,
            name="TemporalCorrelatorNotify",
            daemon=False
        )
        self._notify_thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop the async processing thread.

        Waits for thread to finish current work before returning.
        Safe to call multiple times or before start().
        """
        if not self._started:
            return

        self._shutdown_flag = True
        self._notify_event.set()

        if self._notify_thread is not None:
            self._notify_thread.join(timeout=1.0)
            if self._notify_thread.is_alive():
                print(f"WARNING: {self._notify_thread.name} did not exit cleanly within timeout")

        self._started = False

    def submit(self, buffer_output: BufferOutput) -> None:
        """Submit new buffer data for correlation analysis.

        ONLY stores references to input tensors (~1-5μs).
        Background thread handles GPU computation and callbacks.

        Note: Call start() before first submit() to enable async processing.

        Args:
            buffer_output: Tuple of (values, mask) from RollingFeatureBuffer
        """
        if not self._started:
            raise RuntimeError("TemporalCorrelator not started. Call start() first.")

        # Store references (input tensors are already on GPU and contiguous)
        values, mask = buffer_output
        self._staging_values = values
        self._staging_mask = mask

        # Signal async thread
        self._notify_event.set()

    def _process(self) -> None:
        """Background thread: GPU computation and callback dispatch.

        Runs continuously until shutdown_flag is set.
        Wakes on _notify_event signal from submit().
        """

        while not self._shutdown_flag:
            # Wait for signal from submit()
            self._notify_event.wait()
            self._notify_event.clear()

            if self._shutdown_flag:
                break

            try:
                with torch.no_grad():
                    # Get staged input
                    values = self._staging_values
                    mask = self._staging_mask

                    if values is None or mask is None:
                        continue

                    num_tracks, window_size, feature_length = values.shape

                    # Need at least 2 tracks for pairwise comparison
                    if num_tracks < 2:
                        self._notify_callbacks(SimilarityBatch(similarities=[]))
                        self._frame_count += 1
                        continue

                    # Initialize pair indices on first frame (or if num_tracks changed)
                    expected_pairs = (num_tracks * (num_tracks - 1)) // 2
                    if len(self._pair_indices) != expected_pairs:
                        self._init_pair_indices(num_tracks, values.device)

                    prepare = time.perf_counter()

                    # Step 1: Compute motion energy (RMS over time)
                    energy = self._compute_motion_energy(values, mask)

                    # Step 2: Compute support weights from energy
                    support = self._compute_support_weights(energy)

                    correlate = time.perf_counter()
                    self._T1.add_time((correlate - prepare) * 1000, True)

                    # Step 3: Vectorized bounded-lag correlation for all pairs
                    max_corr = self._compute_all_correlations(values, mask)

                    post = time.perf_counter()
                    self._T2.add_time((post - correlate) * 1000, True)

                    # Step 4: Apply motion gating on GPU (broadcast support)
                    # support: (N, J) -> max_corr: (N, N, J)
                    gated = max_corr * support.unsqueeze(1) * support.unsqueeze(0)

                    # Step 5: Extract unique pairs on GPU using precomputed indices
                    # gated_pairs: (num_pairs, feature_length)
                    gated_pairs = gated[self._pair_i, self._pair_j, :]
                    scores_pairs = (support[self._pair_i, :] * support[self._pair_j, :])

                    # Step 6: Apply EMA on GPU
                    if not self._ema_initialized:
                        self._ema_synchrony = gated_pairs.clone()
                        self._ema_initialized = True
                    else:
                        self._ema_synchrony = (
                            self._ema_alpha * gated_pairs +
                            (1.0 - self._ema_alpha) * self._ema_synchrony
                        )


                    # Step 7: Single GPU->CPU transfer
                    torch.cuda.synchronize()
                    upload = time.perf_counter()
                    self._T3.add_time((upload - post) * 1000, True)
                    gated_np = self._ema_synchrony.cpu().numpy()  # (num_pairs, feature_length)
                    scores_np = scores_pairs.cpu().numpy()  # (num_pairs, feature_length)

                    # Step 8: Build SimilarityFeature list (CPU-only, no GPU ops)
                    similarities: list[SimilarityFeature] = []
                    for idx, pair_id in enumerate(self._pair_indices):
                        similarities.append(SimilarityFeature(
                            pair_id=pair_id,
                            values=gated_np[idx],
                            scores=scores_np[idx]
                        ))

                    finish = time.perf_counter()
                    self._T4.add_time((finish - upload) * 1000, True)

                    # Wrap in batch and notify
                    batch = SimilarityBatch(similarities=similarities)
                    self._notify_callbacks(batch)
                    self._frame_count += 1

            except Exception as e:
                print(f"TemporalCorrelator worker error: {e}")

    def _init_pair_indices(self, num_tracks: int, device: torch.device) -> None:
        """Initialize precomputed pair indices for GPU gather operations.

        Args:
            num_tracks: Number of tracks
            device: GPU device for index tensors
        """
        self._pair_indices = list(combinations(range(num_tracks), 2))

        # GPU tensors for indexing into (N, N, J) correlation matrix
        i_indices = [p[0] for p in self._pair_indices]
        j_indices = [p[1] for p in self._pair_indices]

        self._pair_i = torch.tensor(i_indices, dtype=torch.long, device=device)
        self._pair_j = torch.tensor(j_indices, dtype=torch.long, device=device)

        # Reset EMA when pair structure changes
        self._ema_synchrony = torch.empty(0)
        self._ema_initialized = False

        # Reset pre-allocated buffers (shape will be set in _compute_all_correlations)
        self._x_normalized = torch.empty(0)
        self._corr_max = torch.empty(0)
        self._corr_temp = torch.empty(0)

    def _compute_motion_energy(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute RMS motion energy per joint.

        Args:
            values: (num_tracks, window_size, feature_length) angular velocities
            mask: (num_tracks, window_size, feature_length) validity mask (unused, data is clean)

        Returns:
            energy: (num_tracks, feature_length) RMS energy per joint
        """
        # RMS = sqrt(mean(x^2)) over time dimension
        with torch.no_grad():
            return torch.sqrt(torch.mean(values ** 2, dim=1))

    def _compute_support_weights(self, energy: torch.Tensor) -> torch.Tensor:
        """Convert motion energy to support weights in [0, 1].

        Args:
            energy: (num_tracks, feature_length) RMS energy

        Returns:
            support: (num_tracks, feature_length) weights in [0, 1]
        """

        with torch.no_grad():
            return torch.clamp((energy - self._energy_low) / (self._energy_high - self._energy_low),min=0.0,max=1.0)

    def _compute_all_correlations(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute bounded-lag Pearson correlation for ALL pairs.

        Uses pre-allocated buffers and proper Pearson correlation coefficient formula.

        Args:
            values: (num_tracks, window_size, feature_length) angular velocities
            mask: (num_tracks, window_size, feature_length) validity mask (unused)

        Returns:
            max_corr: (num_tracks, num_tracks, feature_length) max correlation per joint over all lags
        """
        num_tracks, window_size, feature_length = values.shape

        # Allocate buffers on first frame or if shape changed
        expected_shape = (num_tracks, window_size, feature_length)
        if self._x_normalized.shape != expected_shape:
            self._x_normalized = torch.empty(expected_shape, device=values.device)
            self._corr_max = torch.empty((num_tracks, num_tracks, feature_length), device=values.device)
            self._corr_temp = torch.empty((num_tracks, num_tracks, feature_length), device=values.device)

        # 1. Pre-normalize ONCE (center + L2 normalize per joint per track)
        # Reuse self._x_normalized buffer
        mean = values.mean(dim=1, keepdim=True)
        self._x_normalized.copy_(values)
        self._x_normalized.sub_(mean)  # in-place center
        norm = torch.norm(self._x_normalized, dim=1, keepdim=True)
        norm.clamp_(min=self._eps)  # in-place clamp
        self._x_normalized.div_(norm)  # in-place normalize

        # 2. Initialize output buffer
        self._corr_max.fill_(-1.0)

        # 3. Bounded-lag loop (einsum doesn't support out, assign to temp)
        for lag in range(-self._max_lag, self._max_lag + 1):
            y = torch.roll(self._x_normalized, shifts=lag, dims=1)  # (N, W, J)
            self._corr_temp = torch.einsum('nwj,kwj->nkj', self._x_normalized, y)  # (N, K, J)
            torch.maximum(self._corr_max, self._corr_temp, out=self._corr_max)

        # 4. Zero out self-correlation diagonal
        for t in range(num_tracks):
            self._corr_max[t, t, :] = 0.0

        # 5. Clamp to [0, 1] in-place
        torch.clamp(self._corr_max, 0.0, 1.0, out=self._corr_max)

        return self._corr_max