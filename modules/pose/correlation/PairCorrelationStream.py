import math  # ✅ Added
import traceback
import numpy as np
import pandas as pd
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

    def get_top_pairs(self, n: int = 5, duration: Optional[float] = None, min_similarity: Optional[float] = None) -> list[Tuple[int, int]]:
        """
        Get the top N most similar pose pairs based on average similarity scores.

        Args:
            n: Number of top pairs to return
            duration: If provided, only consider data from the last X seconds
            min_similarity: If provided, only include pairs with similarity >= this value

        Returns:
            List of tuples containing (id_1, id_2) sorted by similarity.
            Pairs with all NaN values in the window are excluded.
        """
        result: list[Tuple[int, int, float]] = []

        for pair_id, df in self.pair_history.items():
            if duration is not None and not df.empty:
                stream_end: pd.Timestamp = df.index[-1]
                window_cutoff: pd.Timestamp = stream_end - pd.Timedelta(seconds=duration)
                filtered_df: pd.DataFrame = df[df.index >= window_cutoff]
                if filtered_df.empty:
                    continue
                avg_similarity: float = filtered_df['similarity'].mean(skipna=True)  # ✅ Added skipna
            else:
                avg_similarity = df['similarity'].mean(skipna=True)  # ✅ Added skipna

            # ✅ Skip pairs where all values in window were NaN
            if pd.isna(avg_similarity):
                continue

            if min_similarity is not None and avg_similarity < min_similarity:
                continue

            # Store as (id_1, id_2, avg_similarity)
            result.append((pair_id[0], pair_id[1], avg_similarity))

        # Sort by similarity score (descending) and take top N
        result.sort(key=lambda x: x[2], reverse=True)
        return [(id1, id2) for id1, id2, _ in result[:n]]

    def get_metric_window(self, pair_id: Tuple[int, int], metric_name: str = "similarity", duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the time series data for a specific metric of a pose pair.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            metric_name: The metric to retrieve (default is 'similarity')
            duration: If provided, only return data from the last X seconds

        Returns:
            Numpy array of metric values over time (may contain NaN), or None if pair not found.
            NaN values indicate frames where the metric couldn't be computed.
        """
        if pair_id not in self.pair_history:
            return None

        df: pd.DataFrame = self.pair_history[pair_id]

        if duration is not None:
            stream_end: pd.Timestamp = df.index[-1]
            window_cutoff: pd.Timestamp = stream_end - pd.Timedelta(seconds=duration)
            df = df[df.index >= window_cutoff]

        if df.empty or metric_name not in df.columns:
            return None

        return df[metric_name].to_numpy()  # ✅ Returns array with NaN values

    def get_correlation_for_key(self, index: int, valid_only: bool = False) -> Dict[int, float]:
        """
        Get the correlation values for a specific pose index.

        Args:
            index: The index of the pose to retrieve correlations for
            valid_only: If True, exclude pairs where last similarity is NaN

        Returns:
            Dictionary mapping other_id to similarity_score.
            Values may be NaN if valid_only=False and last sample is NaN.
        """
        correlations: Dict[int, float] = {}
        for pair_id, df in self.pair_history.items():
            if index in pair_id:
                if not df.empty:
                    last_row = df.iloc[-1]
                    other_id = pair_id[1] if pair_id[0] == index else pair_id[0]
                    similarity = last_row['similarity']

                    # ✅ Optionally filter NaN values
                    if valid_only and pd.isna(similarity):
                        continue

                    correlations[other_id] = similarity
        return correlations

    def get_last_value_for(self, x: int, y: int) -> float:
        """Get the last similarity value for a pose pair.

        Args:
            x: First pose ID
            y: Second pose ID

        Returns:
            Last similarity value, or math.nan if pair not found or last value is NaN.
        """
        pair_id = (x, y) if x <= y else (y, x)
        if pair_id in self.pair_history and not self.pair_history[pair_id].empty:
            last_value = self.pair_history[pair_id].iloc[-1]['similarity']
            # ✅ Return the actual value (may be NaN)
            return float(last_value) if not pd.isna(last_value) else math.nan
        return math.nan  # ✅ Changed from 0.0

    def get_valid_sample_count(self, pair_id: Tuple[int, int], duration: Optional[float] = None) -> int:
        """Get the count of non-NaN samples for a pair.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            duration: If provided, only count samples from the last X seconds

        Returns:
            Number of valid (non-NaN) similarity samples
        """
        if pair_id not in self.pair_history:
            return 0

        df: pd.DataFrame = self.pair_history[pair_id]

        if duration is not None and not df.empty:
            stream_end: pd.Timestamp = df.index[-1]
            window_cutoff: pd.Timestamp = stream_end - pd.Timedelta(seconds=duration)
            df = df[df.index >= window_cutoff]

        return int(df['similarity'].notna().sum())

    def get_data_quality(self, pair_id: Tuple[int, int], duration: Optional[float] = None) -> float:
        """Get the ratio of valid samples for a pair.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            duration: If provided, only consider samples from the last X seconds

        Returns:
            Ratio [0.0, 1.0] of valid samples, or 0.0 if pair not found
        """
        if pair_id not in self.pair_history:
            return 0.0

        df: pd.DataFrame = self.pair_history[pair_id]

        if duration is not None and not df.empty:
            stream_end: pd.Timestamp = df.index[-1]
            window_cutoff: pd.Timestamp = stream_end - pd.Timedelta(seconds=duration)
            df = df[df.index >= window_cutoff]

        if df.empty:
            return 0.0

        valid_count = df['similarity'].notna().sum()
        total_count = len(df)

        return float(valid_count / total_count)

    def get_all_qualities(self, duration: Optional[float] = None) -> Dict[Tuple[int, int], float]:
        """Get data quality ratios for all pairs.

        Args:
            duration: If provided, only consider samples from the last X seconds

        Returns:
            Dictionary mapping pair_id to quality ratio [0.0, 1.0]
        """
        return {
            pair_id: self.get_data_quality(pair_id, duration)
            for pair_id in self.pair_history.keys()
        }

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
        """Process a correlation batch and update the pair history.

        Stores all pairs including those with NaN metrics (occlusion, missing data).
        NaN values are preserved in the DataFrame for gap analysis and visualization.
        """
        if batch.is_empty:
            return

        timestamp: pd.Timestamp = batch.timestamp

        # Process each pair correlation in the batch
        for pair_corr in batch.pair_correlations:
            pair_id: Tuple[int, int] = self._get_canonical_pair_id(pair_corr.pair_id)

            # Create a new row with the similarity score and joint correlations
            # ✅ This now stores NaN if geometric_mean is NaN (empty pair)
            new_data: Dict[str, float] = {'similarity': pair_corr.geometric_mean}
            new_data.update(pair_corr.correlations)  # ✅ Per-joint NaN also stored

            # Create a DataFrame with one row
            new_row = pd.DataFrame(new_data, index=[timestamp])

            # Add to existing DataFrame or create a new one
            if pair_id in self._pair_history:
                self._pair_history[pair_id] = pd.concat([self._pair_history[pair_id], new_row])
            else:
                self._pair_history[pair_id] = new_row

            # Ensure the DataFrame is sorted by timestamp
            self._pair_history[pair_id] = self._pair_history[pair_id].sort_index()

        # Prune old data by capacity
        self._prune_by_capacity()

        # Notify callbacks
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
        """Notify all registered callbacks with current data snapshot."""
        # Create snapshot with current timestamp
        if self._pair_history:
            latest_ts: pd.Timestamp = max(df.index[-1] for df in self._pair_history.values() if not df.empty)
        else:
            latest_ts = pd.Timestamp.now()

        data = PairCorrelationStreamData(
            pair_history={k: v.copy() for k, v in self._pair_history.items()},
            timestamp=latest_ts,
            capacity=self.buffer_capacity
        )

        with self._output_lock:
            self._output_data = data

        # Call callbacks with snapshot
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
