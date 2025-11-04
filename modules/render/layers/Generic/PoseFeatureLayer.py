# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules import CaptureDataHub
from modules.gl.Fbo import Fbo
from modules.gl.Mesh import Mesh
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Text import draw_box_string, text_init

from modules.pose.features.PoseAngles import PoseAngleData
from modules.tracker.Tracklet import Tracklet
from modules.pose.Pose import Pose

from modules.pose.features.PosePoints import POSE_COLOR_LEFT, POSE_COLOR_RIGHT
from modules.pose.features.PoseAngleFeatureBase import PoseAngleFeatureBase

from modules.RenderDataHub import RenderDataHub
from modules.CaptureDataHub import CaptureDataHub
from modules.render.meshes.PoseMeshes import PoseMeshes

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.PoseFeature import PoseFeature

class PoseFeatureLayer(LayerBase):
    pose_feature_shader = PoseFeature()

    def __init__(self, data: RenderDataHub, capture_data: CaptureDataHub, cam_id: int) -> None:
        self.data: RenderDataHub = data
        self.capture_data: CaptureDataHub = capture_data
        self.capture_key: str = capture_data.get_unique_consumer_key()
        self.fbo: Fbo = Fbo()
        self.fbo2: Fbo = Fbo()
        self.cam_id: int = cam_id
        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self.fbo2.allocate(width, height, internal_format)
        if not PoseFeatureLayer.pose_feature_shader.allocated:
            PoseFeatureLayer.pose_feature_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.fbo2.deallocate()
        if PoseFeatureLayer.pose_feature_shader.allocated:
            PoseFeatureLayer.pose_feature_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)
        self.fbo2.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # shader gets reset on hot reload, so we need to check if it's allocated
        if not PoseFeatureLayer.pose_feature_shader.allocated:
            PoseFeatureLayer.pose_feature_shader.allocate(monitor_file=True)

        key: int = self.cam_id

        if self.data.get_is_active(key) is False:
            self.fbo.begin()
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)
            self.fbo.end()
            self.fbo2.begin()
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)
            self.fbo2.end()
            return





        values: PoseAngleData = self.data.get_angles(key)
        # range_scale: float = 1.0
        # values: PoseAngleData = self.data.get_velocities(key)
        range_scale: float = 1.0

        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self.capture_data.get_is_active(key):
            raw_pose: Pose | None = self.capture_data.get_raw_pose(key, True, self.capture_key)
            if raw_pose is not None:
                v_c: PoseAngleData = raw_pose.delta_data
                if v_c is not None:
                    PoseFeatureLayer.pose_feature_shader.use(self.fbo.fbo_id, v_c, range_scale, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            smooth_pose: Pose | None = self.capture_data.get_smooth_pose(key, True, self.capture_key)
            if smooth_pose is not None:
                v_c: PoseAngleData = smooth_pose.delta_data
                if v_c is not None:
                    PoseFeatureLayer.pose_feature_shader.use(self.fbo2.fbo_id, v_c, range_scale)


                # Draw joint labels on top of bars
        self.draw_joint_labels(self.fbo2, values)

    @staticmethod
    def draw_joint_labels(fbo: Fbo, feature: PoseAngleFeatureBase) -> None:
        """Draw joint names at the bottom of each bar."""
        num_joints: int = len(feature)
        step: float = fbo.width / num_joints

        fbo.begin()

        # Get joint names from the feature's enum
        joint_enum_type = feature.__class__.joint_enum()
        joint_names: list[str] = [joint_enum_type(i).name for i in range(num_joints)]

        # Alternate colors for readability
        colors: list[tuple[float, float, float, float]] = [
            (*POSE_COLOR_LEFT, 1.0),
            (*POSE_COLOR_RIGHT, 1.0)
        ]

        for i in range(num_joints):
            string: str = joint_names[i]
            x: int = int((i + 0.1) * step)
            y: int = int(fbo.height * 0.5 - 12)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3)) # type: ignore

        fbo.end()