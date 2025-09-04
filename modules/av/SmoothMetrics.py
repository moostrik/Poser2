from dataclasses import dataclass, field, fields, make_dataclass
# from modules.utils.SmoothValue import SmoothValue

from modules.utils.ValueSmoother import ValueSmoother as SmoothValue
from modules.tracker.Tracklet import Tracklet
from modules.pose.PoseDefinitions import Pose, PoseAngleNames, JointAngleDict
from modules.Settings import Settings


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

        # Dynamically create smoothers for all PersonValues fields
        self.smoothers: dict[str, SmoothValue] = {}
        for field_ in fields(PersonValues):
            # print(f"Creating smoother for {field_.name}")
            self.smoothers[field_.name] = SmoothValue(
                alpha = 0.5,
                output_fps=output_fps, 
                is_circular= True if field_.name == "world_angle" else False,
            )
            
        
        self.hot_reloader = HotReloadMethods(self.__class__, True)
            
    def __getattr__(self, name):
        """Allow access to smoothed values as attributes."""
        if name in self.smoothers:
            smoother: SmoothValue = self.smoothers[name]
            value = smoother.get_smoothed_value()
            return value
        raise AttributeError(f"'SmoothPersonValues' object has no attribute '{name}'")
        
    def __setattr__(self, name, value):
        # Avoid recursion for internal attributes
        if "smoothers" in self.__dict__ and name in self.smoothers:
            smoother: SmoothValue = self.smoothers[name]
            smoother.add_value(value)
        else:
            super().__setattr__(name, value)
        
    def add_pose(self, pose: Pose) -> None:
        tracklet: Tracklet | None = pose.tracklet
        if tracklet is not None and tracklet.is_active:
        
            # angles: JointAngleDict | None  = pose.angles
            # if angles is not None:
            #     print(angles)
            #     for angle_name in PoseAngleNames:                
            #         angle_value = angles[angle_name]
            #         setattr(self, angle_name, angle_value)
            approximate_person_length: float = pose.get_approximate_person_length()
            if approximate_person_length is not None:
                self.approximate_person_length = approximate_person_length

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            self.world_angle = world_angle / 360.0

        if pose.is_final:
            self.reset()
            
    def reset(self) -> None:
        """Reset all smoothers"""
        for smoother in self.smoothers.values():
            smoother.reset()  

    def set_alpha(self, value: float) -> None:
        for smoother in self.smoothers.values():
            smoother.alpha = value