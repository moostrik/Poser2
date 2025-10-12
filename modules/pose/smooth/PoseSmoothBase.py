from abc import ABC, abstractmethod

class PoseSmoothBase(ABC):
    @abstractmethod
    def add_pose(self, pose):
        """Add a new pose data point for processing."""
        pass

    @abstractmethod
    def update(self):
        """Update the smoother's internal state."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the smoother to its initial state."""
        pass

    @property
    @abstractmethod
    def is_active(self):
        """Return whether the smoother is active."""
        pass