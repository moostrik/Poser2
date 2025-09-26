# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.pose.PoseDefinitions import Pose, PoseAngleJointNames, PoseJoint
from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class CentreCameraRender(BaseRender):
    def __init__(self, data: DataManager, cam_id: int) -> None:
        self.data: DataManager = data
        self.cam_id: int = cam_id
        self.cam_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.rect_smoother: PoseSmoothRect = PoseSmoothRect()

        # self.last_pose: Pose | None = None
        # self.last_Rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)
        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.cam_fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.cam_fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self.cam_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self.cam_id

        if key != 2:
            return

        pose: Pose | None = self.data.get_pose(key, only_new_data=False, consumer_key=self.key())
        if pose is None: # only on start of program
            return


        if pose.tracklet.is_removed:
            self.rect_smoother.reset()
            self.clear_render()
            return

        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.key())
        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()

        # self.last_Rect = self.rect_smoother._update(pose)
        # print(f"Cam {key} Rect: {self.last_Rect}")



        smooth_rect: Rect = self.rect_smoother.update(pose)



        # print (self.cam_image.width,self.cam_image.height)
        cam_image_aspect_ratio: float = self.cam_image.width / self.cam_image.height

        width: float =  smooth_rect.width / cam_image_aspect_ratio

        x: float =  smooth_rect.x + ( smooth_rect.width - width) / 2.0


        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()

        self.cam_image.draw_roi(0, 0, self.cam_fbo.width, self.cam_fbo.height,
                                x,  smooth_rect.y, width,  smooth_rect.height)

        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.cam_fbo.end()


    def clear_render(self) -> None:
        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.cam_fbo.end()

    def get_fbo(self) -> Fbo:
        return self.cam_fbo





class PoseSmoothRect():
    def __init__(self, aspectratio: float = 9/16, smoothing_factor: float = 0.8) -> None:
        self.src_aspectratio: float = 16 / 9
        self.dst_aspectratio: float = aspectratio

        self.default_rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)
        self.current_rect: Rect | None = None

        # Spring-damper system state
        self.velocity: Rect = Rect(0.0, 0.0, 0.0, 0.0)
        self.spring_constant: float = 1000.0
        self.damping_ratio: float = 0.9

        # Target relative positions
        self.nose_dest_x: float = 0.5
        self.nose_dest_y: float = 0.4
        self.bottom_dest_y: float = 0.95

        hot_reload = HotReloadMethods(self.__class__, True, True)


    def update(self, pose: Pose) -> Rect:
        self.default_rect: Rect = Rect(0.0, 0.0, 0.0, 0.0)

        self.nose_dest_x: float = 0.5
        self.nose_dest_y: float = 0.4
        self.bottom_dest_y: float = 0.95

        self.spring_constant: float = 200.0
        self.damping_ratio: float = 0.9

        # Get the bottom and nose positions
        pose_rect: Rect | None = pose.crop_rect
        if pose_rect is None:
            # print(f"PoseSmoothRect: No crop rect for pose {pose.id}, this should not happen")
            return self.default_rect

        if self.current_rect is None:
            self.current_rect = pose_rect
            return self.current_rect

        bottom: float = min(pose_rect.bottom, 1.0)

        keypoints: np.ndarray | None = pose.absolute_points
        if pose.point_data is None or keypoints is None:
            return self.current_rect
        scores: np.ndarray | None = pose.point_data.scores
        nose_score: float = scores[PoseJoint.nose.value]
        if nose_score < 0.3:
            return self.current_rect
        nose_x: float = keypoints[PoseJoint.nose.value][0]
        nose_y: float = keypoints[PoseJoint.nose.value][1]

        # Calculate rectangle dimensions
        height: float = (bottom - nose_y) / (self.bottom_dest_y- self.nose_dest_y)
        width: float = height * self.dst_aspectratio
        left: float = nose_x - width * self.nose_dest_x
        top: float = nose_y - height * self.nose_dest_y
        new_rect = Rect(left, top, width, height)

        # Apply spring-damper smoothing

        smooth_x, self.velocity.x = self._apply_spring_damper(
            new_rect.x, self.current_rect.x, self.velocity.x,
            self.spring_constant, self.damping_ratio
        )
        smooth_y, self.velocity.y = self._apply_spring_damper(
            new_rect.y, self.current_rect.y, self.velocity.y,
            self.spring_constant, self.damping_ratio
        )
        smooth_h, self.velocity.height = self._apply_spring_damper(
            new_rect.height, self.current_rect.height, self.velocity.height,
            self.spring_constant, self.damping_ratio
        )
        smooth_w: float = smooth_h * self.dst_aspectratio
        self.current_rect = Rect(x=smooth_x, y=smooth_y, height=smooth_h, width=smooth_w)

        return_rect: Rect = self.current_rect

        if pose.tracklet.is_removed:
            self.reset() # redundant

        return return_rect

    def reset(self) -> None:
        self.current_rect = None
        self.velocity = Rect(0.0, 0.0, 0.0, 0.0)

    # STATIC METHODS
    @staticmethod
    def _apply_spring_damper(new_value: float, current_value: float,
                           velocity: float, spring_constant: float,
                           damping_ratio: float, dt: float = 1.0/60.0) -> tuple[float, float]:
        """Apply spring-damper physics to a single value. Returns (new_value, new_velocity)"""
        if current_value is None:
            return new_value, 0.0

        # Calculate spring force
        displacement = new_value - current_value
        spring_force = spring_constant * displacement

        # Calculate damping force
        damping_force = -2.0 * damping_ratio * np.sqrt(spring_constant) * velocity

        # Update velocity
        total_force = spring_force + damping_force
        new_velocity = velocity + total_force * dt

        # Update position
        result_value = current_value + new_velocity * dt

        return result_value, new_velocity