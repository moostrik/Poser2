from dataclasses import dataclass, field, fields, make_dataclass
# from modules.utils.SmoothValue import SmoothValue

from modules.utils.SmoothOneEuro import SmoothOneEuro, SmoothOneEuroCircular
from modules.tracker.Tracklet import Tracklet
from modules.pose.PoseDefinitions import Pose, PoseAngleNames, JointAngleDict
from modules.Settings import Settings

import OneEuroFilter


from modules.utils.HotReloadMethods import HotReloadMethods

# Dynamically create PersonValues dataclass with PoseAngleNames
PersonValues = make_dataclass(
    "PersonValues",
    [('world_angle', float, field(default=0.0)),
     ('approximate_person_length', float, field(default=0.0))] + 
    [(name, float, field(default=0.0)) for name in PoseAngleNames]
)

class SmoothMetrics:
    def __init__(self, settings: Settings):

        input_fps = settings.camera_fps
        output_fps = settings.light_rate
        
        filter = OneEuroFilter.OneEuroFilter(freq=input_fps, mincutoff=1.0, beta=0.0)

        # Dynamically create smoothers for all PersonValues fields
        self.smoothers: dict[str, SmoothOneEuro | SmoothOneEuroCircular] = {}
        for field_ in fields(PersonValues):
            # print(f"Creating smoother for {field_.name}")
            if field_.name == "world_angle":
                self.smoothers[field_.name] = SmoothOneEuroCircular(
                    freq=output_fps, 
                )
            else:
                self.smoothers[field_.name] = SmoothOneEuro(
                    freq=output_fps,
            )
        self.age: float = 0.0

        self.hot_reloader = HotReloadMethods(self.__class__, True)
            
    def __getattr__(self, name):
        """Allow access to smoothed values as attributes."""
        if name in self.smoothers:
            smoother = self.smoothers[name]
            value = smoother.get_smoothed_value()
            return value
        raise AttributeError(f"'SmoothPersonValues' object has no attribute '{name}'")
        
    def __setattr__(self, name, value):
        # Avoid recursion for internal attributes
        if "smoothers" in self.__dict__ and name in self.smoothers:
            smoother = self.smoothers[name]
            smoother.add_value(value)
        else:
            super().__setattr__(name, value)
        
    def update(self) -> None:
        for smoother in self.smoothers.values():
            smoother.update()
        
    def add_pose(self, pose: Pose) -> None:
        tracklet: Tracklet | None = pose.tracklet
        if tracklet is not None and tracklet.is_active and tracklet.age_in_seconds > 2.0:
            self.age = tracklet.age_in_seconds - 2.0  # Start age after 2 seconds of active tracking
        
            # angles: JointAngleDict | None  = pose.angles
            # if angles is not None:
            #     print(angles)
            #     for angle_name in PoseAngleNames:                
            #         angle_value = angles[angle_name]
            #         setattr(self, angle_name, angle_value)
            
            approximate_person_length: float = pose.get_approximate_person_length()
            if approximate_person_length is not None:
                self.approximate_person_length = approximate_person_length #* min(self.age, 1.0)

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            self.world_angle = world_angle / 360.0

        if pose.is_final:
            self.reset()
            
    def reset(self) -> None:
        """Reset all smoothers"""
        for smoother in self.smoothers.values():
            smoother.reset()  

    def set_smoothness(self, value: float) -> None:
        for smoother in self.smoothers.values():
            smoother.set_smoothness(value)
            
    def set_responsiveness(self, value: float) -> None:
        for smoother in self.smoothers.values():
            smoother.set_responsiveness(value)