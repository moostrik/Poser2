"""Motion synchrony analyzer via bounded-lag correlation.

Consumes BufferOutput from RollingFeatureBuffer and computes pairwise motion
synchrony using bounded-lag cross-correlation of joint angular velocities,
weighted by per-joint motion energy and aggregated via harmonic mean.

Outputs SimilarityBatch for unified downstream processing with static similarity.
"""

from dataclasses import dataclass
from itertools import combinations
import threading

import numpy as np
import torch

from modules.ConfigBase import ConfigBase, config_field
from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.similarity.features.SimilarityFeature import SimilarityFeature
from modules.pose.similarity.features.SimilarityBatch import SimilarityBatch

from modules.utils.HotReloadMethods import HotReloadMethods

# Type alias for input (from RollingFeatureBuffer)
BufferOutput = tuple[torch.Tensor, torch.Tensor]


@dataclass
class TemporalCorrelatorConfig(ConfigBase):
    """Configuration for motion synchrony computation.

    Bounded-lag correlation with motion energy weighting.
    """

    max_lag_frames: int = config_field(
        8, min=1, max=60,
        description="Maximum lag in frames for cross-correlation (e.g., 8 frames @ 23fps = 348ms)"
    )
    energy_low: float = config_field(
        0.01, min=0.0, max=10.0,
        description="Motion energy threshold below which joint is considered stationary (rad/s RMS)"
    )
    energy_high: float = config_field(
        0.1, min=0.1, max=20.0,
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

        # Lag range: [-L, ..., 0, ..., +L]
        self._lags = torch.arange(-self._max_lag, self._max_lag + 1, device='cuda')
        self._num_lags = len(self._lags)

        # EMA state per pair - dict keyed by pair_id, initialized lazily
        self._ema_synchrony: dict[tuple[int, int], np.ndarray] = {}

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

                    # Step 1: Compute motion energy (RMS over time)
                    # energy: (num_tracks, feature_length)
                    energy = self._compute_motion_energy(values, mask)

                    # Step 2: Compute support weights from energy
                    # support: (num_tracks, feature_length) in [0, 1]
                    support = self._compute_support_weights(energy)

                    # Step 3: Vectorized bounded-lag correlation for all pairs
                    # max_corr: (num_tracks, num_tracks, feature_length)
                    max_corr = self._compute_all_correlations(values, mask)

                    # Step 4-5: Build SimilarityFeature list with motion gating and EMA
                    similarities: list[SimilarityFeature] = []

                    for i, j in combinations(range(num_tracks), 2):
                        pair_id = (i, j)  # Already ordered since combinations yields i < j

                        # Per-joint correlation for this pair
                        corr_ij = max_corr[i, j]  # (feature_length,)

                        # Motion gating: weight by support
                        gated = corr_ij * support[i] * support[j]

                        # Combined scores (support weights)
                        scores = support[i] * support[j]

                        # Convert to numpy for SimilarityFeature
                        gated_np = gated.cpu().numpy().astype(np.float32)
                        scores_np = scores.cpu().numpy().astype(np.float32)

                        # Apply EMA smoothing per pair
                        if pair_id in self._ema_synchrony:
                            gated_np = (
                                self._ema_alpha * gated_np +
                                (1.0 - self._ema_alpha) * self._ema_synchrony[pair_id]
                            )
                        self._ema_synchrony[pair_id] = gated_np.copy()

                        # Create SimilarityFeature
                        similarities.append(SimilarityFeature(
                            pair_id=pair_id,
                            values=gated_np,
                            scores=scores_np
                        ))

                    # Wrap in batch
                    batch = SimilarityBatch(similarities=similarities)

                    # Sync GPU operations
                    torch.cuda.synchronize()

                    # Notify callbacks with results
                    self._notify_callbacks(batch)
                    self._frame_count += 1

            except Exception as e:
                print(f"TemporalCorrelator worker error: {e}")

    def _compute_motion_energy(self, values: torch.Tensor, mask: torch.Tensor ) -> torch.Tensor:
        """Compute RMS motion energy per joint.

        Args:
            values: (num_tracks, window_size, feature_length) angular velocities
            mask: (num_tracks, window_size, feature_length) validity mask

        Returns:
            energy: (num_tracks, feature_length) RMS energy per joint
        """

        # print("hello")
        # Mask invalid values (contribute 0 to sum)
        masked_values = values * (mask > 0).float()

        # Sum of squares over time
        sum_sq = torch.sum(masked_values ** 2, dim=1)

        # Count valid samples per joint
        valid_count = torch.sum((mask > 0).float(), dim=1).clamp(min=1.0)

        # RMS = sqrt(mean(x^2))
        energy = torch.sqrt(sum_sq / valid_count)

        return energy

    def _compute_support_weights(self, energy: torch.Tensor) -> torch.Tensor:
        """Convert motion energy to support weights in [0, 1].

        Args:
            energy: (num_tracks, feature_length) RMS energy

        Returns:
            support: (num_tracks, feature_length) weights in [0, 1]
        """
        # Linear ramp from energy_low to energy_high
        support = (energy - self._energy_low) / (self._energy_high - self._energy_low + self._eps)
        return torch.clamp(support, 0.0, 1.0)

    def _compute_all_correlations(
        self,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute bounded-lag correlation for ALL pairs vectorized.

        Uses einsum to compute full NxN correlation matrix in one operation.

        Args:
            values: (num_tracks, window_size, feature_length) angular velocities
            mask: (num_tracks, window_size, feature_length) validity mask

        Returns:
            max_corr: (num_tracks, num_tracks, feature_length) max correlation per joint over all lags
        """
        num_tracks, window_size, feature_length = values.shape

        # Normalize velocities for correlation coefficient
        masked_values = values * (mask > 0).float()
        valid_count = torch.sum((mask > 0).float(), dim=1, keepdim=True).clamp(min=1.0)
        mean_val = torch.sum(masked_values, dim=1, keepdim=True) / valid_count
        centered = (masked_values - mean_val) * (mask > 0).float()
        std_val = torch.sqrt(torch.sum(centered ** 2, dim=1, keepdim=True) / valid_count).clamp(min=self._eps)
        normalized = centered / std_val  # (N, W, J)

        # Build all lagged versions: (N, L, W, J)
        lagged = torch.stack([
            torch.roll(normalized, shifts=int(lag), dims=1)
            for lag in self._lags
        ], dim=1)

        lagged_mask = torch.stack([
            torch.roll(mask, shifts=int(lag), dims=1)
            for lag in self._lags
        ], dim=1)  # (N, L, W, J)

        # Combined mask for all pairs: (N_i, N_j, L, W, J)
        # mask_i: (N, 1, 1, W, J), lagged_mask_j: (1, N, L, W, J)
        mask_i = (mask > 0).float().unsqueeze(1).unsqueeze(2)  # (N, 1, 1, W, J)
        mask_j = (lagged_mask > 0).float().unsqueeze(0)  # (1, N, L, W, J)
        combined_mask = mask_i * mask_j  # (N, N, L, W, J)

        # Expand normalized for einsum: (N, 1, 1, W, J)
        norm_i = normalized.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, W, J)
        lagged_j = lagged.unsqueeze(0)  # (1, N, L, W, J)

        # Element-wise product with mask
        products = norm_i * lagged_j * combined_mask  # (N, N, L, W, J)

        # Sum over time dimension
        corr_sum = torch.sum(products, dim=3)  # (N, N, L, J)
        valid_count = torch.sum(combined_mask, dim=3).clamp(min=1.0)  # (N, N, L, J)

        # Normalized correlation
        correlations = corr_sum / valid_count  # (N, N, L, J)

        # Max over lags
        max_corr, _ = torch.max(correlations, dim=2)  # (N, N, J)

        # Clamp to [0, 1]
        return torch.clamp(max_corr, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"TemporalCorrelator("
            f"frames={self._frame_count})"
        )
