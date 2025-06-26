# Standard library imports
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Optional, Callable

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from modules.analysis.JointAngleWindow import JointAngleWindow
from modules.person.Person import Person
from modules.pose.PoseDefinitions import JointAngleDict
from modules.Settings import Settings

from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods

# Type for analysis output callback
AnalysisCallback = Callable[[pd.DataFrame], None]
VisualisationCallback = Callable[[np.ndarray], None]

class AnalysisPipeline(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()

        # Input
        self.person_input_queue: Queue[Person] = Queue()

        # Windowing for joint angles
        self.window_size: int = settings.analysis_window_size
        self.angle_windows: dict[int, pd.DataFrame] = {}
        self.conf_windows: dict[int, pd.DataFrame] = {}

        # Callbacks for analysis output
        self.callback_lock = Lock()
        self.analysis_output_callbacks: set[AnalysisCallback] = set()
        self.visualisation_callbacks: set[VisualisationCallback] = set()

        hot_reloader = HotReloadStaticMethods(self.__class__, True)

    def stop(self) -> None:
        self._stop_event.set()
        with self.callback_lock:
            self.analysis_output_callbacks.clear()
        self.angle_windows.clear()
        self.conf_windows.clear()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                person: Optional[Person] = self.person_input_queue.get(block=True, timeout=0.01)
                if person is not None:
                    try:
                        self._process(person, self._analysis_callback, self.angle_windows, self.conf_windows, self.window_size)
                    except Exception as e:
                        print(f"Error processing person {person.id}: {e}")
                    self.person_input_queue.task_done()
            except Empty:
                continue

    def person_input(self, person: Person) -> None:
        self.person_input_queue.put(person)

    @ staticmethod
    def _process(person: Person, callback: Callable, angles: dict[int, pd.DataFrame], confidences: dict[int, pd.DataFrame], window_size: int) -> None:
        """ Process a person and update the joint angle windows. """
        if person.pose_angles is None:
            return

        # Flatten angles and confidences
        angle_row: dict[str, float] = {}
        conf_row: dict[str, float] = {}
        for k, v in person.pose_angles.items():
            angle_row[f"{k.name}_angle"] = v["angle"]
            conf_row[f"{k.name}_conf"] = v["confidence"]

        # Update angle window
        angle_df = angles.get(person.id, pd.DataFrame())
        angle_df = pd.concat([angle_df, pd.DataFrame([angle_row])], ignore_index=True)
        if len(angle_df) > window_size:
            angle_df = angle_df.iloc[-window_size:]
        angles[person.id] = angle_df

        # Update confidence window
        conf_df = confidences.get(person.id, pd.DataFrame())
        conf_df = pd.concat([conf_df, pd.DataFrame([conf_row])], ignore_index=True)
        if len(conf_df) > window_size:
            conf_df = conf_df.iloc[-window_size:]
        confidences[person.id] = conf_df

        # Interpolate and smooth angles
        angle_df_interp = angle_df.interpolate(method='linear')
        angle_df_smooth = angle_df_interp.rolling(window=5, min_periods=1).mean()

        # Notify callbacks with both DataFrames
        callback(angle_df_smooth, conf_df)

        # print head of the DataFrame for debugging
        # print(f"Processed angles for person {person.id}:\n{angle_df_smooth.head()}")


    def add_analysis_callback(self, callback: AnalysisCallback) -> None:
        """ Register a callback to receive the current pandas DataFrame window. """
        with self.callback_lock:
            self.analysis_output_callbacks.add(callback)

    def _analysis_callback(self, angles: pd.DataFrame, confidences: pd.DataFrame) -> None:
        self._visualisation_callback(angles, confidences)
        """ Handle the output of the analysis. """
        with self.callback_lock:
            for callback in self.analysis_output_callbacks:
                callback(angles)

    def add_visualisation_callback(self, callback: VisualisationCallback) -> None:
        """ Register a callback to receive the visualisation data. """
        with self.callback_lock:
            self.visualisation_callbacks.add(callback)

    def _visualisation_callback(self, angles: pd.DataFrame, confidences: pd.DataFrame) -> None:
        """ Handle the output of the visualisation. """
        angles_np: np.ndarray = np.nan_to_num(angles.to_numpy(), nan=0.5)
        conf_np: np.ndarray = np.nan_to_num(confidences.to_numpy(), nan=0.0)

        # combine angles and confidences into a single array for visualisation
        visualisation_data: np.ndarray = np.concatenate((angles_np, conf_np), axis=1)
        for callback in self.visualisation_callbacks:
            callback(visualisation_data)