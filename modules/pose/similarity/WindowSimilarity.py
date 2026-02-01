# Standard library imports
from dataclasses import dataclass
from itertools import combinations
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
from modules.pose.similarity._utils.WindowTemporal import compute_temporal_synchrony

from modules.utils.PerformanceTimer import PerformanceTimer


# Type alias for window dictionary (track_id -> FeatureWindow)
WindowDict = dict[int, FeatureWindow]


@dataclass
class WindowSimilarityConfig(ConfigBase):
    """Configuration for WindowSimilarity."""
    window_length: int = config_field(30, min=1, max=300, description="Number of frames to compare")
    stride_length: int = config_field(1, min=1, max=30, description="Step size between comparison windows")


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

                # Get input data
                with self._input_lock:
                    windows: WindowDict = self._input_windows

                start_time: float = time.perf_counter()
                # Process similarities
                batch: SimilarityBatch = self._evaluate_window_similarity(windows)
                elapsed_time: float = (time.perf_counter() - start_time) * 1000.0
                self.Timer.add_time(elapsed_time, True)

                # Store and notify
                with self._output_lock:
                    self._output_data = batch
                self._notify_callbacks(batch)

                elapsed_time: float = (time.perf_counter() - start_time) * 1000.0
                # print(f"WindowSimilarity: Processed {len(batch.similarities)} similarities in {elapsed_time:.3f}s")

            except Exception as e:
                print(f"WindowSimilarity: Processing error: {e}")
                traceback.print_exc()

    def _evaluate_window_similarity(self, windows: WindowDict) -> SimilarityBatch:
        """Process all window pairs and compute their similarities."""
        if len(windows) < 2:
            return SimilarityBatch(similarities=[])

        window_length = self._config.window_length

        # Compute similarities for each pair
        similarities: list[SimilarityFeature] = []
        for (id1, window1), (id2, window2) in combinations(windows.items(), 2):
            # Use only the last window_length samples from each window
            values1 = window1.values[-window_length:]
            mask1 = window1.mask[-window_length:]
            values2 = window2.values[-window_length:]
            mask2 = window2.mask[-window_length:]

            # Create sliced FeatureWindow objects
            sliced_window1 = FeatureWindow(
                values=values1,
                mask=mask1,
                feature_enum=window1.feature_enum,
                feature_names=window1.feature_names,
                range=window1.range
            )
            sliced_window2 = FeatureWindow(
                values=values2,
                mask=mask2,
                feature_enum=window2.feature_enum,
                feature_names=window2.feature_names,
                range=window2.range
            )

            # Compute temporal synchrony between the two windows
            values, scores = compute_temporal_synchrony(sliced_window1, sliced_window2)

            # Normalize pair_id ordering
            pair_id: tuple[int, int] = (id1, id2) if id1 <= id2 else (id2, id1)

            similarities.append(SimilarityFeature(
                pair_id=pair_id,
                values=values,
                scores=scores
            ))

        return SimilarityBatch(similarities=similarities)
