import time
import threading
import numpy as np

class FPS:
    def __init__(self, count_samples=60) -> None:
        self.count_samples: int = count_samples
        self.count_processed = -1
        self.count_dropped = 0
        self.deltas: np.ndarray = np.zeros(count_samples, dtype=float)
        self.current_index: int = 0
        self.frame_at: None | float = None
        self.mutex = threading.Lock()

    def set_sample_count(self, sample_count) -> None:
        with self.mutex:
            self.count_samples = sample_count

    def get_rate_minimum(self) -> float:
        with self.mutex:
            if self.count_processed < 1:
                return 0.0
            max_delta: float = 0.0
            if self.count_processed < self.count_samples:
                max_delta = np.max(self.deltas[:self.count_processed])
            else:
                max_delta = np.max(self.deltas)
            return 1000000.0 / max_delta if max_delta > 0.0 else 1000000.0

    def get_rate_maximum(self)-> float:
        with self.mutex:
            if self.count_processed < 1:
                return 0.0
            min_delta: float = 0.0
            if self.count_processed < self.count_samples:
                min_delta = np.min(self.deltas[:self.count_processed])
            else:
                min_delta = np.min(self.deltas)
            return 1000000.0 / min_delta if min_delta > 0.0 else 1000000.0

    def get_rate_average(self) -> float:
        with self.mutex:
            if self.count_processed < 1:
                return 0.0
            avg_delta: float = 0.0
            if self.count_processed < self.count_samples:
                avg_delta = np.sum(self.deltas[:self.count_processed]) / self.count_processed
            else:
                avg_delta = np.sum(self.deltas) / self.deltas.size
            return 1000000.0 / avg_delta if avg_delta > 0.0 else 1000000.0

    def get_count_processed(self) -> int:
        return self.count_processed

    def get_count_dropped(self) -> int:
        return self.count_dropped

    def processed(self) -> None:
        with self.mutex:
            last_frame_at: None | float = self.frame_at
            self.frame_at = time.time()
            self.count_processed += 1

            if last_frame_at is not None:
                delta: float = (self.frame_at - last_frame_at) * 1000000.0
                # print(delta)
                self.deltas[self.current_index] = delta
                self.current_index = (self.current_index + 1) % self.count_samples  # Update the index circularly

    def dropped(self) -> None:
        self.count_dropped += 1

    def reset(self) -> None:
        with self.mutex:
            self.frame_at = None
            self.count_dropped = 0
            self.count_processed = -1