# Standard library imports
from collections import deque
from typing import Optional

# Third-party imports
import pandas as pd

# Local application imports
from modules.pose.PoseDefinitions import JointAngleDict, Keypoint

class JointAngleWindow:
    def __init__(self, window_size: int = 200):
        """
        Args:
            window_size: Number of time steps to keep in the window
        """
        self.window_size = window_size
        self.data = deque(maxlen=window_size)  # stores dicts
        self.df: Optional[pd.DataFrame] = None

    def add(self, joint_angle_dict: JointAngleDict):
        """
        Add a new JointAngleDict (one time step) to the window.
        """
        # Flatten dict for DataFrame: {Keypoint.left_elbow: {'angle':..., 'confidence':...}, ...}
        flat = {}
        for k, v in joint_angle_dict.items():
            flat[f"{k.name}_angle"] = v["angle"]
            flat[f"{k.name}_conf"] = v["confidence"]
        self.data.append(flat)
        self.df = pd.DataFrame(self.data)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the current window as a pandas DataFrame.
        """
        if self.df is None:
            return pd.DataFrame()
        return self.df.copy()

    def clear(self):
        """
        Clear the window.
        """
        self.data.clear()
        self.df = None