# Standard library imports
from dataclasses import dataclass
import threading
import time
import traceback
from typing import Optional

# Third-party imports
import numpy as np

# Pose imports
from modules.ConfigBase import ConfigBase, config_field
from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.nodes.windows.WindowNode import FeatureWindow
from modules.pose.similarity.features.SimilarityFeature import SimilarityFeature
from modules.pose.similarity.features.SimilarityBatch import SimilarityBatch
from modules.utils.PerformanceTimer import PerformanceTimer

from modules.utils.HotReloadMethods import HotReloadMethods

# Type alias for window dictionary (track_id -> FeatureWindow)
WindowDict = dict[int, FeatureWindow]


@dataclass
class WindowSimilarityConfig(ConfigBase):
    """Configuration for WindowSimilarity."""
    window_length: int = config_field(30, min=1, max=300, description="Number of frames to compare")
    exponent: float = config_field(3.5, min=0.5, max=4.0, description="Similarity decay exponent")


class WindowSimilarity(TypedCallbackMixin[SimilarityBatch]):
    """Computes pairwise window similarities in a background thread.

    Processes FeatureWindow data from active tracklets and computes similarity metrics
    for all pairs based on temporal patterns. Results are published via callbacks.

    Unlike FrameSimilarity which compares single frames, WindowSimilarity compares
    temporal sequences (windows) using time-series similarity algorithms like DTW,
    cross-correlation, or other sequence matching methods.
    """

    def __init__(self, config: WindowSimilarityConfig | None = None) -> None:
        super().__init__()

        self._config = config if config is not None else WindowSimilarityConfig()

        self._correlation_thread = threading.Thread(target=self.run, daemon=True)
        self._stop_event = threading.Event()

        # INPUTS - Just store the latest data with a lock
        self._input_lock = threading.Lock()
        self._input_windows: WindowDict = {}
        self._update_event = threading.Event()

        # OUTPUT
        self._output_lock = threading.Lock()
        self._output_data: Optional[SimilarityBatch] = None

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

                # Process similarity
                start_time: float = time.perf_counter()
                batch: SimilarityBatch = self._process(windows)
                elapsed_time: float = (time.perf_counter() - start_time) * 1000.0
                self.Timer.add_time(elapsed_time, True)

                # Store and notify
                with self._output_lock:
                    self._output_data = batch
                self._notify_callbacks(batch)

            except Exception as e:
                print(f"WindowSimilarity: Processing error: {e}")
                traceback.print_exc()

    def _process(self, windows: WindowDict) -> SimilarityBatch:
        if len(windows) < 2:
            return SimilarityBatch(similarities=[])

        # Step 1: Stack windows
        track_ids, values = self._stack_windows(windows)
        # Step 2: Compute similarity tensor
        best_sim, confidence_scores, leader_scores = self._compute_similarity_tensor(values)
        # Step 3: Build similarity features
        batch = self._build_similarity_batch(track_ids, best_sim, confidence_scores)

        # Debug: Print all pairs (both directions) with all metrics
        N = len(track_ids)

        results = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    id_i = track_ids[i]
                    id_j = track_ids[j]

                    # Harmonic mean (only valid joints, replace near-zero with TINY)
                    _TINY = 1e-5
                    valid = confidence_scores[i, j] > 0
                    if np.any(valid):
                        safe_values = np.where(best_sim[i, j][valid] > _TINY, best_sim[i, j][valid], _TINY)
                        sim = len(safe_values) / np.sum(1.0 / safe_values)
                    else:
                        sim = 0.0
                    conf = np.mean(confidence_scores[i, j])
                    lead = leader_scores[i, j]
                    results.append(f"({id_i},{id_j}):sim={sim:.3f},conf={conf:.3f},lead={lead:+.2f}")

        print(f"Pairs: {' | '.join(results)}")

        return batch


    def _stack_windows(self, windows: WindowDict) -> tuple[list[int], np.ndarray]:
        """Stack all windows into (N, T, F) tensor with padding.

        Returns:
            track_ids: List of track IDs
            values: Stacked window values (N, T, F)
        """
        window_length = self._config.window_length

        track_ids: list[int] = []
        slices: list[np.ndarray] = []

        for track_id, window in windows.items():
            track_ids.append(track_id)

            # Get last window_length samples
            window_data = window.values[-window_length:]

            # Pad with NaN if shorter than window_length
            if len(window_data) < window_length:
                pad_len = window_length - len(window_data)
                padding = np.full((pad_len, window.feature_len), np.nan, dtype=np.float32)
                window_data = np.vstack([padding, window_data])

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
        with np.errstate(invalid='ignore'):
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

    def _build_similarity_batch(
        self,
        track_ids: list[int],
        best_sim: np.ndarray,
        confidence_scores: np.ndarray
    ) -> SimilarityBatch:
        """Build SimilarityBatch from similarity tensor.

        Args:
            track_ids: List of track IDs
            best_sim: Best similarity per joint (N, N, F)
            confidence_scores: Confidence scores (N, N, F)

        Returns:
            SimilarityBatch with all pairwise similarities
        """
        N = len(track_ids)
        i_idx, j_idx = np.triu_indices(N, k=1)

        similarities: list[SimilarityFeature] = []
        for i, j in zip(i_idx, j_idx):
            id1, id2 = track_ids[i], track_ids[j]
            pair_id = (id1, id2) if id1 <= id2 else (id2, id1)

            similarities.append(SimilarityFeature(
                pair_id=pair_id,
                values=best_sim[i, j],
                scores=confidence_scores[i, j]
            ))

        return SimilarityBatch(similarities=similarities)
