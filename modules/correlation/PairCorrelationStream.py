import numpy as np
import pandas as pd
from dataclasses import dataclass
from multiprocessing import Process, Queue, Event
from threading import Thread
from typing import Callable, Optional, Dict, Tuple, Set

from modules.correlation.PairCorrelation import PairCorrelationBatch
from modules.Settings import Settings
from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass(frozen=True)
class PairCorrelationStreamInput:
    batch: PairCorrelationBatch

@dataclass(frozen=True)
class PairCorrelationStreamData:
    pair_history: Dict[Tuple[int, int], pd.DataFrame]
    timestamp: pd.Timestamp

    def get_top_pairs(self, n: int = 5, duration: Optional[float] = None, min_similarity: Optional[float] = None) -> list[Tuple[int, int]]:
        """
        Get the top N most similar pose pairs based on average similarity scores.

        Args:
            n: Number of top pairs to return
            duration: If provided, only consider data from the last X seconds
            min_similarity: If provided, only include pairs with similarity >= this value

        Returns:
            List of tuples containing (id_1, id_2) sorted by similarity
        """
        result: list[Tuple[int, int, float]] = []

        for pair_id, df in self.pair_history.items():
            if duration is not None and not df.empty:
                stream_end: pd.Timestamp = df.index[-1]
                window_cutoff: pd.Timestamp = stream_end - pd.Timedelta(seconds=duration)
                filtered_df: pd.DataFrame = df[df.index >= window_cutoff]
                if filtered_df.empty:
                    continue
                avg_similarity: float = filtered_df['similarity'].mean()
            else:
                avg_similarity = df['similarity'].mean()

            if min_similarity is not None and avg_similarity < min_similarity:
                continue

            # Store as (id_1, id_2, avg_similarity)
            result.append((pair_id[0], pair_id[1], avg_similarity))

        # Sort by similarity score (descending) and take top N
        result.sort(key=lambda x: x[2], reverse=True)
        return [(id1, id2) for id1, id2, _ in result[:n]]

    def get_metric_window(self, pair_id: Tuple[int, int], metric_name: str = "similarity", duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the time series data for a specific joint of a pose pair.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            metric_name: The metric to retrieve (default is 'similarity')
            duration: If provided, only return data from the last X seconds

        Returns:
            Numpy array of joint values over time, or None if the pair is not found
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

        return df[metric_name].to_numpy()

PairCorrelationStreamDataCallback = Callable[[PairCorrelationStreamData], None]

class PairCorrelationStreamManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings  # Store settings for processor recreation
        self.result_queue = Queue()
        self.processor = PairCorrelationStreamProcessor(settings, self.result_queue)
        self.output_callbacks: Set[PairCorrelationStreamDataCallback] = set()
        self.result_thread = Thread(target=self._handle_results, daemon=True)
        self.running = False

        # Hot reload setup for restarting processor
        self.hot_reloader = HotReloadMethods(PairCorrelationStreamProcessor, True, True)
        self.hot_reloader.add_file_changed_callback(self._on_file_changed)

    def _on_file_changed(self) -> None:
        """Restart the processor when files change."""
        print("[PairCorrelationStream] File changed, restarting processor...")
        if self.running:
            self._restart_processor()

    def _restart_processor(self) -> None:
        """Restart the processor process."""
        try:
            # Stop current processor
            if self.processor.is_alive():
                print("[PairCorrelationStream] Stopping current processor...")
                self.processor.stop()
                self.processor.join(timeout=2.0)  # Wait a bit longer for graceful shutdown

                if self.processor.is_alive():
                    print("[PairCorrelationStream] Force terminating processor...")
                    self.processor.terminate()
                    self.processor.join(timeout=1.0)

            # Create new processor
            print("[PairCorrelationStream] Creating new processor...")
            self.processor = PairCorrelationStreamProcessor(self.settings, self.result_queue)
            self.processor.start()
            print("[PairCorrelationStream] Processor restarted successfully")

        except Exception as e:
            print(f"[PairCorrelationStream] Error restarting processor: {e}")

    def start(self) -> None:
        """Start the processor and result handler."""
        self.running = True
        self.processor.start()
        self.result_thread.start()

    def stop(self) -> None:
        """Stop the processor and result handler."""
        self.running = False

        # Stop hot reloader
        self.hot_reloader.stop_file_watcher()

        # Stop processor
        self.processor.stop()
        self.processor.join(timeout=1.0)
        if self.processor.is_alive():
            self.processor.terminate()

    def add_correlation(self, batch: PairCorrelationBatch) -> None:
        """Add correlation batch to processing queue."""
        try:
            correlation_input = PairCorrelationStreamInput(batch=batch)
            self.processor.add_correlation(correlation_input)
        except Exception as e:
            print(f"[PairCorrelationStream] Error adding correlation: {e}")

    def add_stream_callback(self, callback: PairCorrelationStreamDataCallback) -> None:
        """Register a callback to receive processed data."""
        self.output_callbacks.add(callback)

    def _handle_results(self) -> None:
        """Handle results from the processor in the main process."""
        while self.running:
            try:
                data: PairCorrelationStreamData = self.result_queue.get(timeout=0.1)
                # Call all callbacks with the data
                for callback in self.output_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"Error in callback: {e}")
            except:
                continue

class PairCorrelationStreamProcessor(Process):
    def __init__(self, settings: Settings, result_queue: Queue, history_duration: float = 60.0) -> None:
        super().__init__()

        # Use multiprocessing primitives
        self._stop_event = Event()
        self.correlation_input_queue: Queue[PairCorrelationStreamInput] = Queue()

        # For sending results back to main process
        self.result_queue = result_queue if result_queue else Queue()

        # Store settings values
        self.history_duration = pd.Timedelta(seconds=history_duration)

        # Initialize pair history (will be recreated in child process)
        self._pair_history: Dict[Tuple[int, int], pd.DataFrame] = {}

    def stop(self) -> None:
        self._stop_event.set()
        self._pair_history.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                correlation_input: Optional[PairCorrelationStreamInput] = self.correlation_input_queue.get(block=True, timeout=0.01)
                if correlation_input is not None:
                    try:
                        self._process(correlation_input)
                    except Exception as e:
                        print(f"Error processing correlation: {e}")
            except:
                continue

    def add_correlation(self, correlation_input: PairCorrelationStreamInput) -> None:
        """Add correlation to processing queue - can be called from main process."""
        try:
            self.correlation_input_queue.put(correlation_input, block=False)
        except:
            # Queue is full, skip this correlation
            pass

    def _notify_callbacks(self, data: PairCorrelationStreamData) -> None:
        """ Send results back to main process via queue. """
        try:
            self.result_queue.put(data, block=False)
        except:
            pass

    def _process(self, correlation_input: PairCorrelationStreamInput) -> None:
        """Process a correlation batch and update the pair history."""
        batch = correlation_input.batch

        if batch.is_empty:
            return

        timestamp: pd.Timestamp = batch.timestamp
        current_time: pd.Timestamp = pd.Timestamp.now()
        cutoff_time: pd.Timestamp = current_time - self.history_duration

        # Process each pair correlation in the batch
        for pair_corr in batch.pair_correlations:
            pair_id: Tuple[int, int] = self.get_canonical_pair_id(pair_corr.pair_id)

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

        # Send results back to main process
        self._notify_callbacks(self.get_stream_data())

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

    def get_stream_data(self) -> PairCorrelationStreamData:
        """Return a snapshot of the current pair correlation stream data."""
        # Use the latest timestamp among all pairs, or pd.Timestamp.now() if empty
        if self._pair_history:
            latest_ts: pd.Timestamp = max(df.index[-1] for df in self._pair_history.values() if not df.empty)
        else:
            latest_ts = pd.Timestamp.now()
        return PairCorrelationStreamData(
            pair_history={k: v.copy() for k, v in self._pair_history.items()},
            timestamp=latest_ts
        )

    @staticmethod
    def get_canonical_pair_id(pair_id: Tuple[int, int]) -> Tuple[int, int]:
        """
        Create a canonical pair ID by ensuring the smaller ID is always first.
        """
        return (pair_id[0], pair_id[1]) if pair_id[0] <= pair_id[1] else (pair_id[1], pair_id[0])
