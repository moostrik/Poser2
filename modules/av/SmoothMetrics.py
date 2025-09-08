from dataclasses import dataclass, field, fields, make_dataclass
# from modules.utils.SmoothValue import SmoothValue

import numpy as np

from modules.utils.SmoothOneEuro import SmoothOneEuro, SmoothOneEuroCircular
from modules.tracker.Tracklet import Tracklet
from modules.pose.PoseDefinitions import Pose, PoseAngleNames, JointAngleDict, PoseAngleKeypoints
from modules.Settings import Settings

import OneEuroFilter


from modules.utils.HotReloadMethods import HotReloadMethods

class SmoothMetrics:
    def __init__(self, settings: Settings):
        # object.__setattr__(self, "filters", {})
        # object.__setattr__(self, "age", 0.0)

        # input_fps = settings.camera_fps
        output_fps = settings.light_rate
        
        self.filters: dict[str, SmoothOneEuro | SmoothOneEuroCircular] = {}
        
        self.filters["world_angle"] = SmoothOneEuroCircular(freq = output_fps)
        self.filters["approximate_person_length"] = SmoothOneEuro(freq = output_fps)
        for key in PoseAngleKeypoints.keys():
            angle_name = key.name
            self.filters[angle_name] = SmoothOneEuro(freq = output_fps)

        self.age: float = 0.0

        self.hot_reloader = HotReloadMethods(self.__class__, True)
            
    def __getattr__(self, name):
        """Allow access to smoothed values as attributes."""
        if "filters" in self.__dict__ and name in self.filters:
            smoother = self.filters[name]
            value = smoother.get_smoothed_value()
            return value
        raise AttributeError(f"'SmoothMetrics' object has no attribute '{name}'")
        
    def __setattr__(self, name, value):
        # Avoid recursion for internal attributes
        if "filters" in self.__dict__ and name in self.filters:
            filter = self.filters[name]
            filter.add_value(value)
        else:
            super().__setattr__(name, value)
        
    def update(self) -> None:
        for filter in self.filters.values():
            filter.update()

    def add_pose(self, pose: Pose) -> None:
        tracklet: Tracklet | None = pose.tracklet
        if tracklet is not None and tracklet.is_active and tracklet.age_in_seconds > 2.0:
            self.age = tracklet.age_in_seconds - 2.0  # Start age after 2 seconds of active tracking
        
            angles: JointAngleDict | None  = pose.angles
            if angles is not None:
                # print(angles)
                for key in PoseAngleKeypoints.keys():                
                    angle_value = angles[key]
                    angle_name = key.name
                    if angle_value['confidence'] > 0.3 and angle_value['angle'] is not np.nan:
                        setattr(self, angle_name, angle_value['angle'])

                    # print(f"Setting {angle_name} to {angle_value.angle}")
                    # setattr(self, angle_name, angle_value)

            approximate_person_length: float = pose.get_approximate_person_length()
            if approximate_person_length is not None:
                self.approximate_person_length = approximate_person_length #* min(self.age, 1.0)

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            # convert from [0...360] to [-pi, pi]
            self.world_angle = np.deg2rad(world_angle - 180)
            # print (self.world_angle)

        if pose.is_final:
            self.reset()
            
    def reset(self) -> None:
        """Reset all smoothers"""
        for smoother in self.filters.values():
            smoother.reset()  

    def set_smoothness(self, value: float) -> None:
        for smoother in self.filters.values():
            smoother.set_smoothness(value)
            
    def set_responsiveness(self, value: float) -> None:
        for smoother in self.filters.values():
            smoother.set_responsiveness(value)