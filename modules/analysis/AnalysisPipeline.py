# Standard library imports
from threading import Lock
from typing import Optional, Callable

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from modules.person.Person import Person
from modules.pose.PoseDefinitions import JointAngleDict
from modules.analysis.JointAngleWindow import JointAngleWindow

# Type for analysis output callback
AnalysisCallback = Callable[[pd.DataFrame], None]

class AnalysisPipeline:
    def __init__(self, window_size: int = 200) -> None:
        # Windowing for joint angles
        self.joint_angle_window = JointAngleWindow(window_size=window_size)
        # Callbacks for analysis output
        self.callback_lock = Lock()
        self.analysis_output_callbacks: set[AnalysisCallback] = set()
        self.running: bool = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
        with self.callback_lock:
            self.analysis_output_callbacks.clear()
        self.joint_angle_window.clear()

    def person_input(self, person: Person) -> None:
        """
        Accepts a Person object, extracts joint angles, and updates the window.
        Calls registered callbacks with the updated DataFrame.
        """
        if hasattr(person, "pose_angles") and person.pose_angles is not None:
            angle: Optional[JointAngleDict] = person.pose_angles
            self.joint_angle_window.add(person.pose_angles)
            df: pd.DataFrame = self.joint_angle_window.get_dataframe()
            with self.callback_lock:
                for callback in self.analysis_output_callbacks:
                    callback(df)

    def add_analysis_callback(self, callback: AnalysisCallback) -> None:
        """
        Register a callback to receive the current pandas DataFrame window.
        """
        if self.running:
            print('AnalysisPipeline is running, cannot add callback')
            return
        self.analysis_output_callbacks.add(callback)

    def get_window(self, as_numpy: bool = False) -> pd.DataFrame | np.ndarray:
        """
        Get the current window as a pandas DataFrame or numpy array.
        """
        df: pd.DataFrame = self.joint_angle_window.get_dataframe()
        if as_numpy:
            return df.to_numpy()
        return df