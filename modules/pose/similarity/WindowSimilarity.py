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
        best_sim, scores = self._compute_similarity_tensor(values)
        # Step 3: Build similarity features
        batch = self._build_similarity_batch(track_ids, best_sim, scores)

        # Debug: Print aggregate similarity for each pair in one line
        # pairs_str = " | ".join([f"{sim.pair_id}:{sim.harmonic_mean():.3f}" for sim in batch])
        # print(f"Similarities: {pairs_str}")

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

    def _compute_similarity_tensor(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute pairwise similarity tensor using vectorized operations.

        Args:
            values: Stacked window values (N, T, F)

        Returns:
            best_sim: Best similarity per joint (N, N, F)
            scores: Confidence scores (N, N, F)
        """
        exponent = self._config.exponent

        # Broadcast to compute all pairwise angular differences
        # values[:, None, :, None, :] -> (N, 1, T, 1, F)
        # values[None, :, None, :, :] -> (1, N, 1, T, F)
        # Result: (N, N, T, T, F) - all person pairs, all time pairs, all features
        raw_diff = values[:, None, :, None, :] - values[None, :, None, :, :]

        # Wrap angular difference to [-π, π] and compute similarity
        angular_diff = np.mod(raw_diff + np.pi, 2 * np.pi) - np.pi
        similarity = np.power(1.0 - np.abs(angular_diff) / np.pi, exponent)
        # NaN propagates automatically through all operations

        # Best per joint via nanmax over time axes
        # (N, N, T, T, F) -> (N, N, F)
        best_sim = np.nanmax(similarity, axis=(2, 3))
        best_sim = np.nan_to_num(best_sim, nan=0.0).astype(np.float32)

        # Scores from valid comparison proportion
        scores = 1.0 - np.mean(np.isnan(similarity), axis=(2, 3))
        scores = scores.astype(np.float32)

        return best_sim, scores

    def _build_similarity_batch(
        self,
        track_ids: list[int],
        best_sim: np.ndarray,
        scores: np.ndarray
    ) -> SimilarityBatch:
        """Build SimilarityBatch from similarity tensor.

        Args:
            track_ids: List of track IDs
            best_sim: Best similarity per joint (N, N, F)
            scores: Confidence scores (N, N, F)

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
                scores=scores[i, j]
            ))

        return SimilarityBatch(similarities=similarities)
