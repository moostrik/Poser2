"""CUDA-accelerated motion synchrony analyzer with custom kernel.

Drop-in replacement for TemporalCorrelator using custom CUDA kernel for
bounded-lag correlation. Falls back to PyTorch implementation if CUDA
unavailable or compilation fails.
"""

from dataclasses import dataclass
from itertools import combinations
import threading
import time
import os
from pathlib import Path
from typing import Any

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


# CUDA kernel source - TWO KERNELS: normalize + correlate
CUDA_KERNEL_SOURCE = """
#include <cuda_runtime.h>

// Kernel 1: Normalize values in-place (center + L2 normalize per track per joint)
// Grid: (N, J), Block: (1)
// Each thread handles one (track, joint) combination
extern "C" __global__
void normalize_kernel(
    float* __restrict__ values,  // (N, W, J) - modified in-place
    const int N,
    const int W,
    const int J,
    const float eps
) {
    const int track = blockIdx.x;
    const int joint = blockIdx.y;

    if (track >= N || joint >= J) return;

    // Compute mean
    float sum = 0.0f;
    for (int w = 0; w < W; w++) {
        sum += values[track * W * J + w * J + joint];
    }
    const float mean = sum / W;

    // Center and compute L2 norm
    float norm_sq = 0.0f;
    for (int w = 0; w < W; w++) {
        const int idx = track * W * J + w * J + joint;
        const float centered = values[idx] - mean;
        values[idx] = centered;
        norm_sq += centered * centered;
    }

    // Normalize
    const float norm = sqrtf(fmaxf(norm_sq, eps));
    for (int w = 0; w < W; w++) {
        const int idx = track * W * J + w * J + joint;
        values[idx] /= norm;
    }
}

// Kernel 2: Bounded-lag correlation on pre-normalized data
// Grid: (N, N), Block: (J) where J <= 1024
// Each thread handles one (i, j, joint) combination
// Uses streaming loads - no local arrays
extern "C" __global__
void correlation_kernel(
    const float* __restrict__ values,  // (N, W, J) - already normalized
    float* __restrict__ max_corr,      // (N, N, J) output
    const int N,
    const int W,
    const int J,
    const int max_lag
) {
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    const int joint = threadIdx.x;

    if (i >= N || j >= N || joint >= J) return;

    // Self-correlation = 0
    if (i == j) {
        max_corr[i * N * J + j * J + joint] = 0.0f;
        return;
    }

    // Base pointers for this joint
    const int base_i = i * W * J + joint;
    const int base_j = j * W * J + joint;

    // Compute max correlation over all lags (streaming loads, no local arrays)
    float max_c = -1.0f;

    for (int lag = -max_lag; lag <= max_lag; lag++) {
        float dot = 0.0f;

        for (int w = 0; w < W; w++) {
            // Circular shift with single modulo
            const int w_shifted = (w + lag + W) % W;

            // Streaming loads - registers only
            const float xi = values[base_i + w * J];
            const float xj = values[base_j + w_shifted * J];
            dot += xi * xj;
        }

        max_c = fmaxf(max_c, dot);
    }

    // Clamp to [0, 1] and write
    max_corr[i * N * J + j * J + joint] = fmaxf(0.0f, fminf(1.0f, max_c));
}
"""


@dataclass
class TemporalCorrelatorConfig(ConfigBase):
    """Configuration for CUDA-accelerated motion synchrony computation."""

    max_lag_frames: int = config_field(
        15, min=1, max=60,
        description="Maximum lag in frames for cross-correlation"
    )
    energy_low: float = config_field(
        0.1, min=0.0, max=10.0,
        description="Motion energy threshold for stationary joints (rad/s RMS)"
    )
    energy_high: float = config_field(
        0.5, min=0.1, max=20.0,
        description="Motion energy threshold for full contribution (rad/s RMS)"
    )
    ema_alpha: float = config_field(
        0.8, min=0.0, max=1.0,
        description="EMA smoothing factor"
    )
    eps: float = config_field(
        1e-6, min=1e-10, max=1e-3,
        description="Epsilon for numerical stability"
    )
    use_cuda_kernel: bool = config_field(
        True,
        description="Use custom CUDA kernel (falls back to PyTorch if unavailable)"
    )


class TemporalCorrelator(TypedCallbackMixin[SimilarityBatch]):
    """CUDA-accelerated motion synchrony analyzer with custom kernel.

    Same interface as TemporalCorrelator but uses custom CUDA kernel for
    2-3x speedup on correlation computation.

    Falls back to PyTorch implementation if:
    - CUDA not available
    - Kernel compilation fails
    - use_cuda_kernel=False in config
    """

    def __init__(self, config: TemporalCorrelatorConfig) -> None:
        super().__init__()

        self._config = config
        self._max_lag = config.max_lag_frames
        self._energy_low = config.energy_low
        self._energy_high = config.energy_high
        self._ema_alpha = config.ema_alpha
        self._eps = config.eps

        # EMA state
        self._ema_synchrony: torch.Tensor = torch.empty(0)
        self._ema_initialized: bool = False

        # Precomputed pair indices
        self._pair_indices: list[tuple[int, int]] = []
        self._pair_i: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._pair_j: torch.Tensor = torch.empty(0, dtype=torch.long)

        # Pre-allocated buffers for PyTorch fallback
        self._x_normalized: torch.Tensor = torch.empty(0)
        self._corr_max: torch.Tensor = torch.empty(0)
        self._corr_temp: torch.Tensor = torch.empty(0)

        # Staging references
        self._staging_values: torch.Tensor | None = None
        self._staging_mask: torch.Tensor | None = None

        # Async notification
        self._notify_event = threading.Event()
        self._shutdown_flag = False
        self._started = False
        self._notify_thread: threading.Thread | None = None
        self._frame_count = 0

        # Performance timers
        self._T1: PerformanceTimer = PerformanceTimer(name="prepare  ", sample_count=200, report_interval=100, color="red", omit_init=25)
        self._T2: PerformanceTimer = PerformanceTimer(name="correlate", sample_count=200, report_interval=100, color="green", omit_init=25)
        self._T3: PerformanceTimer = PerformanceTimer(name="finish   ", sample_count=200, report_interval=100, color="blue", omit_init=25)
        self._T4: PerformanceTimer = PerformanceTimer(name="download ", sample_count=200, report_interval=100, color="yellow", omit_init=25)

        # Try to load/compile CUDA kernels
        self._normalize_kernel: Any = None
        self._correlation_kernel: Any = None
        self._use_cuda_kernel = False

        if config.use_cuda_kernel and torch.cuda.is_available():
            try:
                self._normalize_kernel, self._correlation_kernel = self._compile_cuda_kernels()
                self._use_cuda_kernel = True
                print("TemporalCorrelatorCUDA: Custom CUDA kernels loaded successfully")
            except Exception as e:
                print(f"TemporalCorrelatorCUDA: CUDA kernel compilation failed, using PyTorch fallback: {e}")
        else:
            print("TemporalCorrelatorCUDA: Using PyTorch fallback (CUDA disabled or unavailable)")

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def _compile_cuda_kernels(self) -> tuple[Any, Any]:
        """Compile CUDA kernels using torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import load_inline

        # Compile kernels
        cuda_module = load_inline(
            name='bounded_lag_correlation_v2',
            cpp_sources='',
            cuda_sources=CUDA_KERNEL_SOURCE,
            functions=['normalize_kernel', 'correlation_kernel'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )

        if cuda_module is None:
            raise RuntimeError("Failed to compile CUDA kernels")

        normalize_kernel = getattr(cuda_module, 'normalize_kernel', None)
        correlation_kernel = getattr(cuda_module, 'correlation_kernel', None)

        if normalize_kernel is None or correlation_kernel is None:
            raise RuntimeError("Compiled module missing required kernel functions")

        return normalize_kernel, correlation_kernel

    def start(self) -> None:
        """Start async processing thread."""
        if self._started:
            return

        self._shutdown_flag = False
        self._notify_thread = threading.Thread(
            target=self._process,
            name="TemporalCorrelatorCUDANotify",
            daemon=False
        )
        self._notify_thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop async processing thread."""
        if not self._started:
            return

        self._shutdown_flag = True
        self._notify_event.set()

        if self._notify_thread is not None:
            self._notify_thread.join(timeout=1.0)
            if self._notify_thread.is_alive():
                print(f"WARNING: {self._notify_thread.name} did not exit cleanly")

        self._started = False

    def submit(self, buffer_output: BufferOutput) -> None:
        """Submit buffer data for analysis."""
        if not self._started:
            raise RuntimeError("TemporalCorrelatorCUDA not started. Call start() first.")

        values, mask = buffer_output
        self._staging_values = values
        self._staging_mask = mask
        self._notify_event.set()

    def _process(self) -> None:
        """Background thread: GPU computation and callback dispatch."""
        while not self._shutdown_flag:
            self._notify_event.wait()
            self._notify_event.clear()

            if self._shutdown_flag:
                break

            try:
                with torch.no_grad():
                    values = self._staging_values
                    mask = self._staging_mask

                    if values is None or mask is None:
                        continue

                    num_tracks, window_size, feature_length = values.shape

                    if num_tracks < 2:
                        self._notify_callbacks(SimilarityBatch(similarities=[]))
                        self._frame_count += 1
                        continue

                    expected_pairs = (num_tracks * (num_tracks - 1)) // 2
                    if len(self._pair_indices) != expected_pairs:
                        self._init_pair_indices(num_tracks, values.device)

                    prepare = time.perf_counter()

                    # Step 1-2: Energy and support (same as PyTorch version)
                    energy = self._compute_motion_energy(values, mask)
                    support = self._compute_support_weights(energy)

                    correlate = time.perf_counter()
                    self._T1.add_time((correlate - prepare) * 1000, True)

                    # Step 3: Correlation (CUDA kernel or PyTorch)
                    if self._use_cuda_kernel:
                        max_corr = self._compute_correlations_cuda(values)
                    else:
                        max_corr = self._compute_correlations_pytorch(values, mask)

                    post = time.perf_counter()
                    self._T2.add_time((post - correlate) * 1000, True)

                    # Step 4-6: Motion gating, pair extraction, EMA (same as PyTorch)
                    gated = max_corr * support.unsqueeze(1) * support.unsqueeze(0)
                    gated_pairs = gated[self._pair_i, self._pair_j, :]
                    scores_pairs = support[self._pair_i, :] * support[self._pair_j, :]

                    if not self._ema_initialized:
                        self._ema_synchrony = gated_pairs.clone()
                        self._ema_initialized = True
                    else:
                        self._ema_synchrony = (
                            self._ema_alpha * gated_pairs +
                            (1.0 - self._ema_alpha) * self._ema_synchrony
                        )

                    # Step 7-8: Download and build batch
                    torch.cuda.synchronize()
                    upload = time.perf_counter()
                    self._T3.add_time((upload - post) * 1000, True)

                    gated_np = self._ema_synchrony.cpu().numpy()
                    scores_np = scores_pairs.cpu().numpy()

                    similarities: list[SimilarityFeature] = []
                    for idx, pair_id in enumerate(self._pair_indices):
                        similarities.append(SimilarityFeature(
                            pair_id=pair_id,
                            values=gated_np[idx],
                            scores=scores_np[idx]
                        ))

                    finish = time.perf_counter()
                    self._T4.add_time((finish - upload) * 1000, True)

                    batch = SimilarityBatch(similarities=similarities)
                    self._notify_callbacks(batch)
                    self._frame_count += 1

            except Exception as e:
                print(f"TemporalCorrelatorCUDA worker error: {e}")
                import traceback
                traceback.print_exc()

    def _compute_correlations_cuda(self, values: torch.Tensor) -> torch.Tensor:
        """Compute correlations using custom CUDA kernels.

        Two-kernel approach:
        1. normalize_kernel: Center + L2 normalize (O(N * W * J))
        2. correlation_kernel: Bounded-lag dot products (O(N² * W * J * L))

        Args:
            values: (N, W, J) angular velocities

        Returns:
            max_corr: (N, N, J) max correlation over lags
        """
        num_tracks, window_size, feature_length = values.shape
        stream = torch.cuda.current_stream().cuda_stream

        # Allocate normalized buffer if needed (will be modified in-place by kernel)
        if self._x_normalized.shape != values.shape:
            self._x_normalized = torch.empty_like(values)

        # Copy values to normalized buffer (kernel modifies in-place)
        self._x_normalized.copy_(values)

        # Allocate output if needed
        if self._corr_max.shape != (num_tracks, num_tracks, feature_length):
            self._corr_max = torch.empty(
                (num_tracks, num_tracks, feature_length),
                device=values.device,
                dtype=torch.float32
            )

        if self._normalize_kernel is None or self._correlation_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")

        # Kernel 1: Normalize - grid (N, J), block (1)
        normalize_grid = (num_tracks, feature_length, 1)
        normalize_block = (1, 1, 1)

        self._normalize_kernel(
            normalize_grid,
            normalize_block,
            [
                self._x_normalized.data_ptr(),
                num_tracks,
                window_size,
                feature_length,
                self._eps
            ]
        )

        # Kernel 2: Correlation - grid (N, N), block (J)
        corr_grid = (num_tracks, num_tracks, 1)
        corr_block = (feature_length, 1, 1)

        self._correlation_kernel(
            corr_grid,
            corr_block,
            [
                self._x_normalized.data_ptr(),
                self._corr_max.data_ptr(),
                num_tracks,
                window_size,
                feature_length,
                self._max_lag
            ]
        )

        return self._corr_max

    def _compute_correlations_pytorch(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback for correlation computation (same as TemporalCorrelator)."""
        num_tracks, window_size, feature_length = values.shape

        # Allocate buffers if needed
        expected_shape = (num_tracks, window_size, feature_length)
        if self._x_normalized.shape != expected_shape:
            self._x_normalized = torch.empty(expected_shape, device=values.device)
            self._corr_max = torch.empty((num_tracks, num_tracks, feature_length), device=values.device)
            self._corr_temp = torch.empty((num_tracks, num_tracks, feature_length), device=values.device)

        # Pre-normalize
        mean = values.mean(dim=1, keepdim=True)
        self._x_normalized.copy_(values)
        self._x_normalized.sub_(mean)
        norm = torch.norm(self._x_normalized, dim=1, keepdim=True)
        norm.clamp_(min=self._eps)
        self._x_normalized.div_(norm)

        # Initialize output
        self._corr_max.fill_(-1.0)

        # Bounded-lag loop
        for lag in range(-self._max_lag, self._max_lag + 1):
            y = torch.roll(self._x_normalized, shifts=lag, dims=1)
            self._corr_temp = torch.einsum('nwj,kwj->nkj', self._x_normalized, y)
            torch.maximum(self._corr_max, self._corr_temp, out=self._corr_max)

        # Zero diagonal
        for t in range(num_tracks):
            self._corr_max[t, t, :] = 0.0

        # Clamp
        torch.clamp(self._corr_max, 0.0, 1.0, out=self._corr_max)

        return self._corr_max

    def _init_pair_indices(self, num_tracks: int, device: torch.device) -> None:
        """Initialize pair indices."""
        self._pair_indices = list(combinations(range(num_tracks), 2))

        i_indices = [p[0] for p in self._pair_indices]
        j_indices = [p[1] for p in self._pair_indices]

        self._pair_i = torch.tensor(i_indices, dtype=torch.long, device=device)
        self._pair_j = torch.tensor(j_indices, dtype=torch.long, device=device)

        self._ema_synchrony = torch.empty(0)
        self._ema_initialized = False

        self._x_normalized = torch.empty(0)
        self._corr_max = torch.empty(0)
        self._corr_temp = torch.empty(0)

    def _compute_motion_energy(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute RMS motion energy per joint."""
        return torch.sqrt(torch.mean(values ** 2, dim=1))

    def _compute_support_weights(self, energy: torch.Tensor) -> torch.Tensor:
        """Convert energy to support weights [0, 1]."""
        return torch.clamp(
            (energy - self._energy_low) / (self._energy_high - self._energy_low),
            min=0.0,
            max=1.0
        )