import numpy as np
import pandas as pd
from threading import Lock
from typing import Callable, Optional, Dict, Tuple

from modules.Settings import Settings

class PoseCorrelation:
    def __init__(self, id_1: int, id_2: int, joint_correlations: dict[str, float]) -> None:
        self.pair_id: Tuple[int, int] = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
        self.joint_correlations: dict[str, float] = joint_correlations
        self.similarity_score: float = float(np.mean(list(joint_correlations.values()))) if joint_correlations else 0.0

class PoseCorrelationBatch:
    def __init__(self) -> None:
        """Collection of DTW results from a single analysis run."""
        self._pair_correlations: list[PoseCorrelation] = []
        self._timestamp: pd.Timestamp = pd.Timestamp.now()
        self._similarity: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Check if the batch has no results."""
        return len(self._pair_correlations) == 0

    @property
    def count(self) -> int:
        """Return the number of windows in the batch."""
        return len(self._pair_correlations)

    @property
    def timestamp(self) -> pd.Timestamp:
        return self._timestamp

    @property
    def similarity(self) -> float:
        return self._similarity

    def add_result(self, result: PoseCorrelation) -> None:
        """Add a PoseCorrelation result to the batch."""
        self._pair_correlations.append(result)
        self._similarity = sum(r.similarity_score for r in self._pair_correlations) / len(self._pair_correlations)

        # Sort by similarity score
        self._pair_correlations.sort(key=lambda r: r.similarity_score, reverse=True)

    def get_most_similar_pair(self) -> Optional[PoseCorrelation]:
        """Return the pair with highest similarity score."""
        if not self._pair_correlations:
            return None
        return max(self._pair_correlations, key=lambda r: r.similarity_score)

PoseCorrelationBatchCallback = Callable[[PoseCorrelationBatch], None]


# PoseCorrelationWindowCallback = Callable[[Dict[Tuple[int, int], pd.DataFrame]], None]

class PoseCorrelationWindow:
    def __init__(self, settings: Settings, history_duration: float = 60.0) -> None:
        """
        Tracks pose correlation batches over time in a rolling window using pandas DataFrames.

        Args:
            history_duration: Duration in seconds to maintain history
        """
        self.history_duration = pd.Timedelta(seconds=history_duration)
        self._pair_history: Dict[Tuple[int, int], pd.DataFrame] = {}

        self._update_callbacks_lock = Lock()
        self._update_callbacks: set[Callable[['PoseCorrelationWindow'], None]] = set()

    def add_batch(self, batch: PoseCorrelationBatch) -> None:
        """
        Add a new batch to the history by extracting individual pair correlations.

        Args:
            batch: The PoseCorrelationBatch to process
        """
        if not batch.is_empty:

            timestamp: pd.Timestamp = batch.timestamp
            current_time: pd.Timestamp = pd.Timestamp.now()
            cutoff_time: pd.Timestamp = current_time - self.history_duration

            # Process each pair correlation in the batch
            for pair_corr in batch._pair_correlations:
                pair_id: Tuple[int, int] = pair_corr.pair_id

                # Create a new row with the similarity score and joint correlations
                new_data: Dict[str, float] = {'similarity': pair_corr.similarity_score}
                new_data.update(pair_corr.joint_correlations)

                # Create a DataFrame with one row
                new_row = pd.DataFrame(new_data, index=[timestamp])

                # Add to existing DataFrame or create a new one
                if pair_id in self._pair_history:
                    self._pair_history[pair_id] = pd.concat([self._pair_history[pair_id], new_row])
                else:
                    self._pair_history[pair_id] = new_row

                # Ensure the DataFrame is sorted by timestamp
                self._pair_history[pair_id] = self._pair_history[pair_id].sort_index()

            # Prune old data
            self._prune_history(cutoff_time)

            self._notify_update_callbacks()


    def _prune_history(self, cutoff_time: pd.Timestamp) -> None:
        """
        Remove data points older than the cutoff time from all pairs.

        Args:
            cutoff_time: Timestamp before which data should be removed
        """
        pairs_to_remove: list[Tuple[int,int]] = []

        for pair_id, df in self._pair_history.items():
            # Keep only rows with timestamps >= cutoff_time
            pruned_df: pd.DataFrame = df[df.index >= cutoff_time]

            if pruned_df.empty:
                # All data is old, mark for removal
                pairs_to_remove.append(pair_id)
            else:
                # Update with pruned data
                self._pair_history[pair_id] = pruned_df

        # Remove pairs with no remaining data
        for pair_id in pairs_to_remove:
            del self._pair_history[pair_id]

    def get_top_pairs(self, n: int = 5, time_window: Optional[float] = None, min_similarity: Optional[float] = None) -> list[Tuple[int, int]]:
        """
        Get the top N most similar pose pairs based on average similarity scores.

        Args:
            n: Number of top pairs to return
            time_window: If provided, only consider data from the last X seconds
            min_similarity: If provided, only include pairs with similarity >= this value

        Returns:
            List of tuples containing (id_1, id_2) sorted by similarity
        """
        result: list[Tuple[int, int, float]] = []
        current_time: pd.Timestamp = pd.Timestamp.now()

        for pair_id, df in self._pair_history.items():
            # Apply time window filter if specified
            if time_window is not None:
                window_cutoff: pd.Timestamp = current_time - pd.Timedelta(seconds=time_window)
                filtered_df: pd.DataFrame = df[df.index >= window_cutoff]

                # Skip if no data in the window
                if filtered_df.empty:
                    continue

                avg_similarity: float = filtered_df['similarity'].mean()
            else:
                avg_similarity = df['similarity'].mean()

            # Skip if below minimum similarity threshold
            if min_similarity is not None and avg_similarity < min_similarity:
                continue

            # Store as (id_1, id_2, avg_similarity)
            result.append((pair_id[0], pair_id[1], avg_similarity))

        # Sort by similarity score (descending) and take top N
        result.sort(key=lambda x: x[2], reverse=True)
        return [(id1, id2) for id1, id2, _ in result[:n]]

    def get_metric_window(self, pair_id: Tuple[int, int], metric_name: str = "similarity'", time_window: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the time series data for a specific joint of a pose pair.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            joint: The joint to retrieve (default is 'similarity')
            time_window: If provided, only return data from the last X seconds

        Returns:
            Numpy array of joint values over time, or None if the pair is not found
        """
        pair_id = self.get_canonical_pair_id(pair_id)
        if pair_id not in self._pair_history:
            return None

        df: pd.DataFrame = self._pair_history[pair_id]

        # Apply time window filter if specified
        if time_window is not None:
            window_cutoff = pd.Timestamp.now() - pd.Timedelta(seconds=time_window)
            df = df[df.index >= window_cutoff]

        # Return the joint values as a numpy array
        return df[metric_name].to_numpy() if metric_name in df.columns else None

    def get_dataframe(self, pair_id: Tuple[int,int], time_window: Optional[float] = None, resample: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get the correlation data for a specific pair.

        Args:
            id_1: First pose ID
            id_2: Second pose ID
            time_window: If provided, only return data from the last X seconds
            resample: If provided, resample data to the specified frequency (e.g., '1s', '500ms')

        Returns:
            DataFrame with timestamps as index and columns for similarity and joints,
            or None if the pair is not found
        """
        pair_id = self.get_canonical_pair_id(pair_id)
        if pair_id not in self._pair_history:
            return None

        df: pd.DataFrame = self._pair_history[pair_id]

        # Apply time window filter if specified
        if time_window is not None:
            window_cutoff = pd.Timestamp.now() - pd.Timedelta(seconds=time_window)
            df = df[df.index >= window_cutoff]

        # Apply resampling if specified
        if resample is not None and not df.empty:
            # Resample to the specified frequency
            # For downsampling, use mean aggregation
            df = df.resample(resample).mean()

        return df

    def get_metric_statistics(self, pair_id: Tuple[int,int], metric_name: Optional[str] = None, time_window: Optional[float] = None) -> Optional[Dict]:
        """
        Get statistics for a specific metric or all metrics for a pair.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            metric_name: If provided, only return stats for this metric (joint name or 'similarity')
            time_window: If provided, only consider data from the last X seconds

        Returns:
            Dictionary with statistics (mean, std, min, max) for the specified metric(s)
            or None if the pair is not found
        """
        pair_id = self.get_canonical_pair_id(pair_id)
        df: Optional[pd.DataFrame] = self.get_dataframe(pair_id, time_window)

        if df is None or df.empty:
            return None

        if metric_name is not None:
            # Check if the metric exists
            if metric_name not in df.columns:
                return None

            # Return statistics for a specific metric
            stats = {
                'mean': df[metric_name].mean(),
                'std': df[metric_name].std(),
                'min': df[metric_name].min(),
                'max': df[metric_name].max()
            }
            return stats
        else:
            # Return statistics for all metrics (joints and similarity)
            return df.describe().to_dict()

    @staticmethod
    def get_canonical_pair_id(pair_id: Tuple[int, int]) -> Tuple[int, int]:
        """
        Create a canonical pair ID by ensuring the smaller ID is always first.
        """
        return (pair_id[0], pair_id[1]) if pair_id[0] <= pair_id[1] else (pair_id[1], pair_id[0])

    def add_update_callback(self, callback: Callable[['PoseCorrelationWindow'], None]) -> None:
        """ Register a callback to receive the current pandas DataFrame window. """
        with self._update_callbacks_lock:
            self._update_callbacks.add(callback)

    def _notify_update_callbacks(self) -> None:
        """ Call all registered callbacks with the current batch. """
        with self._update_callbacks_lock:
            for callback in self._update_callbacks:
                callback(self)
