import time
import threading

class FPS:
    def __init__(self, count_samples=60):
        self.count_samples = count_samples
        self.count_processed = 0
        self.count_dropped = 0
        self.deltas = []
        self.frame_at = None
        self.mutex = threading.Lock()

    def set_sample_count(self, sample_count):
        with self.mutex:
            self.count_samples = sample_count

    def get_rate_minimum(self):
        with self.mutex:
            if not self.deltas:
                return 0.0
            max_delta = max(self.deltas)
            return 1000000.0 / max_delta if max_delta > 0.0 else 1000000.0

    def get_rate_maximum(self):
        with self.mutex:
            if not self.deltas:
                return 0.0
            min_delta = min(self.deltas)
            return 1000000.0 / min_delta if min_delta > 0.0 else 1000000.0

    def get_rate_average(self):
        with self.mutex:
            if not self.deltas:
                return 0.0
            avg_delta = sum(self.deltas) / len(self.deltas)
            return 1000000.0 / avg_delta if avg_delta > 0.0 else 1000000.0

    def get_count_processed(self):
        return self.count_processed

    def get_count_dropped(self):
        return self.count_dropped

    def processed(self):
        with self.mutex:
            last_frame_at = self.frame_at
            self.frame_at = time.time()
            self.count_processed += 1

            if last_frame_at is not None:
                delta = (self.frame_at - last_frame_at) * 1000000.0
                self.deltas.append(delta)

                if len(self.deltas) > self.count_samples:
                    trim = len(self.deltas) - self.count_samples
                    self.deltas = self.deltas[trim:]

    def dropped(self):
        self.count_dropped += 1

    def reset(self):
        with self.mutex:
            self.frame_at = None
            self.count_dropped = 0
            self.count_processed = 0
            self.deltas.clear()