import math
import traceback
import numpy as np
import pandas as pd
from functools import cached_property
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Event, Lock
from typing import Callable, Optional, Dict, Tuple, Set

from modules.pose.correlation.PairCorrelation import PairCorrelationBatch

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass(frozen=True)
class PairCorrelationStreamData:
    """Immutable snapshot of pose pair correlation history.

    Stores time-series data for each pose pair including similarity metrics
    and per-joint correlations. NaN values indicate frames where the pair
    couldn't be compared (occlusion, missing detection, etc).

    Attributes:
        pair_history: Dictionary mapping pair IDs to their DataFrame history.
                     DataFrames may contain NaN values for similarity and joints.
        timestamp: Timestamp when this snapshot was created
        capacity: Maximum number of samples stored per pair
    """
    pair_history: Dict[Tuple[int, int], pd.DataFrame]
    timestamp: pd.Timestamp
    capacity: int

    @cached_property
    def _sorted_pairs_by_similarity(self) -> list[Tuple[Tuple[int, int], float]]:
        """Pre-compute sorted pairs by average similarity (descending).

        Returns:
            List of ((id_1, id_2), avg_similarity) tuples, sorted by similarity.
            Pairs with all NaN values are excluded.
        """
        result: list[Tuple[Tuple[int, int], float]] = []

        for pair_id, df in self.pair_history.items():
            avg_similarity = df['similarity'].mean(skipna=True)
            if not pd.isna(avg_similarity):
                result.append((pair_id, avg_similarity))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_top_pairs(self, n: int = 5) -> list[Tuple[int, int]]:
        """
        Get the top N most similar pose pairs based on average similarity scores.

        Args:
            n: Number of top pairs to return

        Returns:
            List of tuples containing (id_1, id_2) sorted by similarity.
            Pairs with all NaN values in the window are excluded.
        """
        # Fast path: use cached result if no duration/filter specified
        return [pair_id for pair_id, _ in self._sorted_pairs_by_similarity[:n]]

    def get_metric_window_array(self, pair_id: Tuple[int, int], metric_name: str = "similarity",
                                 max_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get fixed-size metric array padded/truncated to max_samples.
        Optimized for rendering where array size must be consistent.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            metric_name: The metric to retrieve (default is 'similarity')
            max_samples: Fixed array size. If None, returns full history.

        Returns:
            Numpy array of metric values (may contain NaN).
            If max_samples specified, array is right-aligned (most recent data).
        """
        if pair_id not in self.pair_history:
            return None

        df: pd.DataFrame = self.pair_history[pair_id]

        if df.empty or metric_name not in df.columns:
            return None

        values: np.ndarray = df[metric_name].to_numpy()

        if max_samples is None or len(values) <= max_samples:
            return values

        # Return most recent samples
        return values[-max_samples:]

    @cached_property
    def _last_similarities(self) -> Dict[Tuple[int, int], float]:
        """Pre-compute last similarity values for all pairs."""
        return {
            pair_id: (df.iloc[-1]['similarity'] if not df.empty else math.nan)
            for pair_id, df in self.pair_history.items()
        }

    def get_last_value_for(self, x: int, y: int) -> float:
        """Get the last similarity value for a pose pair.

        Args:
            x: First pose ID
            y: Second pose ID

        Returns:
            Last similarity value, or math.nan if pair not found or last value is NaN.
        """
        pair_id = (x, y) if x <= y else (y, x)
        value = self._last_similarities.get(pair_id, math.nan)
        return float(value) if not pd.isna(value) else math.nan

    @cached_property
    def _precomputed_windows(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Pre-compute all similarity windows once per snapshot."""
        result = {}
        for pair_id, df in self.pair_history.items():
            if not df.empty and 'similarity' in df.columns:
                result[pair_id] = df['similarity'].to_numpy()
        return result

    def get_top_pairs_with_windows(self, n: int, max_samples: int,
                                   metric_name: str = "similarity") -> list[Tuple[Tuple[int, int], np.ndarray]]:
        """
        Combined method optimized for rendering: get top pairs with their metric arrays.
        """
        top_pairs = self.get_top_pairs(n)
        result: list[Tuple[Tuple[int, int], np.ndarray]] = []

        # ✅ Use pre-computed windows when requesting similarity
        if metric_name == "similarity":
            for pair_id in top_pairs:
                if pair_id in self._precomputed_windows:
                    values = self._precomputed_windows[pair_id]
                    if len(values) > max_samples:
                        values = values[-max_samples:]
                    result.append((pair_id, values))
        else:
            # Slow path for other metrics
            for pair_id in top_pairs:
                if pair_id in self.pair_history:
                    df = self.pair_history[pair_id]
                    if metric_name in df.columns:
                        values = df[metric_name].to_numpy()
                        if len(values) > max_samples:
                            values = values[-max_samples:]
                        result.append((pair_id, values))

        return result

PairCorrelationStreamDataCallback = Callable[[PairCorrelationStreamData], None]


class PairCorrelationStream(Thread):
    def __init__(self, capacity: int, timeout: float) -> None:
        super().__init__(daemon=True)

        # Thread synchronization
        self._stop_event = Event()
        self._input_queue: Queue[PairCorrelationBatch] = Queue()
        self._output_lock = Lock()

        # Callbacks for sending data
        self._data_callbacks: Set[PairCorrelationStreamDataCallback] = set()
        self._output_data: Optional[PairCorrelationStreamData] = None

        # Store settings values
        self.buffer_capacity: int = capacity
        self.timeout: float = timeout

        # Initialize pair history
        self._pair_history: Dict[Tuple[int, int], pd.DataFrame] = {}

        self.hot_reload = HotReloadMethods(self.__class__, True, True)

    def stop(self) -> None:
        """Stop the processing thread."""
        self._stop_event.set()
        self.join()

    def run(self) -> None:
        """Main worker thread loop."""
        while not self._stop_event.is_set():
            try:
                batch: PairCorrelationBatch = self._input_queue.get(timeout=0.01)
                try:
                    self._process_batch(batch)
                except Exception as e:
                    print(f"[PairCorrelationStream] Error processing batch: {e}")
                    traceback.print_exc()
            except Empty:
                pass

            self._prune_by_timeout()

        self._pair_history.clear()

    def _process_batch(self, batch: PairCorrelationBatch) -> None:
        """Process a correlation batch - OPTIMIZED for similarity only."""
        if batch.is_empty:
            return

        timestamp: pd.Timestamp = batch.timestamp

        for pair_corr in batch.pair_correlations:
            pair_id: Tuple[int, int] = self._get_canonical_pair_id(pair_corr.pair_id)

            # ✅ Only store similarity - massive memory/speed improvement
            new_data: Dict[str, float] = {'similarity': pair_corr.geometric_mean}
            # ❌ REMOVED: new_data.update(pair_corr.correlations)

            new_row = pd.DataFrame(new_data, index=[timestamp])

            if pair_id in self._pair_history:
                self._pair_history[pair_id] = pd.concat([self._pair_history[pair_id], new_row])
            else:
                self._pair_history[pair_id] = new_row

        # ✅ Only prune periodically, not every batch
        self._batch_count = getattr(self, '_batch_count', 0) + 1
        if self._batch_count % 10 == 0:
            self._prune_by_capacity()

        self._notify_callbacks()

    def _prune_by_capacity(self) -> None:
        """
        Remove oldest entries exceeding max_capacity from all pairs.
        """
        pairs_to_remove: list[Tuple[int, int]] = []

        for pair_id, df in self._pair_history.items():
            pruned_df: pd.DataFrame = df.iloc[-self.buffer_capacity:]

            if pruned_df.empty:
                pairs_to_remove.append(pair_id)
            else:
                self._pair_history[pair_id] = pruned_df

        # Remove pairs with no remaining data
        for pair_id in pairs_to_remove:
            del self._pair_history[pair_id]

    def _prune_by_timeout(self) -> None:
        """Remove pairs that have not been updated within the timeout period."""
        pairs_to_remove: list[Tuple[int, int]] = []

        for pair_id, history in self._pair_history.items():
            if not history.empty:
                last_update: pd.Timestamp = history.index[-1]
                if (pd.Timestamp.now() - last_update).total_seconds() > self.timeout:
                    pairs_to_remove.append(pair_id)

        for pair_id in pairs_to_remove:
            del self._pair_history[pair_id]

        # Notify if any pairs were removed
        if pairs_to_remove:
            self._notify_callbacks()

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks - OPTIMIZED."""
        if self._pair_history:
            latest_ts: pd.Timestamp = max(df.index[-1] for df in self._pair_history.values() if not df.empty)
        else:
            latest_ts = pd.Timestamp.now()

        # ✅ Use view() instead of copy() if callbacks are read-only
        # Or implement copy-on-write semantics
        data = PairCorrelationStreamData(
            pair_history=self._pair_history.copy(),  # Shallow copy of dict
            timestamp=latest_ts,
            capacity=self.buffer_capacity
        )

        with self._output_lock:
            self._output_data = data

        for callback in self._data_callbacks.copy():
            try:
                callback(data)
            except Exception as e:
                print(f"[PairCorrelationStream] Error in callback: {e}")
                traceback.print_exc()

    def add_correlation(self, batch: PairCorrelationBatch) -> None:
        """Add correlation batch to processing queue."""
        try:
            self._input_queue.put(batch, block=False)
        except Exception as e:
            print(f"[PairCorrelationStream] Error adding correlation: {e}")

    def add_stream_callback(self, callback: PairCorrelationStreamDataCallback) -> None:
        """Register a callback to receive processed data."""
        self._data_callbacks.add(callback)

    def get_stream_data(self) -> PairCorrelationStreamData | None:
        """Get the latest processed stream data."""
        with self._output_lock:
            return self._output_data

    @staticmethod
    def _get_canonical_pair_id(pair_id: Tuple[int, int]) -> Tuple[int, int]:
        """Create a canonical pair ID by ensuring the smaller ID is always first."""
        return (pair_id[0], pair_id[1]) if pair_id[0] <= pair_id[1] else (pair_id[1], pair_id[0])
