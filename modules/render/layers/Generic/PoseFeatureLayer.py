# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Mesh import Mesh
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Text import draw_box_string, text_init

from modules.pose.features.AngleFeature import AngleFeature
from modules.tracker.Tracklet import Tracklet
from modules.pose.Pose import Pose

from modules.pose.features.deprecated.PoseVertices import POSE_COLOR_LEFT, POSE_COLOR_RIGHT
from modules.pose.features import AngleFeature

from modules.data.RenderDataHub import RenderDataHub
from modules.data.CaptureDataHub import CaptureDataHub
from modules.render.meshes.PoseMeshesCapture import PoseMeshesCapture

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.PoseFeature import PoseFeature

class PoseFeatureLayer(LayerBase):
    pose_feature_shader = PoseFeature()

    def __init__(self, render_data: RenderDataHub, capture_data: CaptureDataHub, cam_id: int) -> None:
        self.render_data: RenderDataHub = render_data
        self.capture_data: CaptureDataHub = capture_data
        self.capture_key: str = capture_data.get_unique_consumer_key()
        self.raw_fbo: Fbo = Fbo()
        self.smooth_fbo: Fbo = Fbo()
        self.render_fbo: Fbo = Fbo()
        self.cam_id: int = cam_id
        self.draw_raw: bool = True
        self.draw_smooth: bool = True
        self.draw_render: bool = True

        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.raw_fbo.allocate(width, height, internal_format)
        self.smooth_fbo.allocate(width, height, internal_format)
        self.render_fbo.allocate(width, height, internal_format)
        if not PoseFeatureLayer.pose_feature_shader.allocated:
            PoseFeatureLayer.pose_feature_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.raw_fbo.deallocate()
        self.smooth_fbo.deallocate()
        self.render_fbo.deallocate()
        if PoseFeatureLayer.pose_feature_shader.allocated:
            PoseFeatureLayer.pose_feature_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        if self.draw_raw:
            self.raw_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        if self.draw_smooth:
            self.smooth_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        if self.draw_render:
            self.render_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # shader gets reset on hot reload, so we need to check if it's allocated
        if not PoseFeatureLayer.pose_feature_shader.allocated:
            PoseFeatureLayer.pose_feature_shader.allocate(monitor_file=True)

        key: int = self.cam_id

        if self.render_data.is_active(key) is False:
            self.raw_fbo.clear()
            self.smooth_fbo.clear()
            self.render_fbo.clear()
            return


        range_scale: float = 1.0

        LayerBase.setView(self.raw_fbo.width, self.raw_fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        raw_color = (0.0, 0.0, 0.0)
        smooth_color = (0.0, 0.5, 0.5)
        render_color = (0.5, 0.0, 0.0)
        self.draw_raw = True
        self.draw_smooth = False
        self.draw_render = True

        if self.capture_data.get_is_active(key):
            raw_pose: Pose | None = self.capture_data.get_raw_pose(key, True, self.capture_key)
            if raw_pose is not None:
                self.raw_fbo.clear()
                data: AngleFeature = raw_pose.angle_data
                PoseFeatureLayer.pose_feature_shader.use(self.raw_fbo.fbo_id, data, range_scale, raw_color, raw_color)
            smooth_pose: Pose | None = self.capture_data.get_smooth_pose(key, True, self.capture_key)
            if smooth_pose is not None:
                self.smooth_fbo.clear()
                data: AngleFeature = smooth_pose.angle_data
                PoseFeatureLayer.pose_feature_shader.use(self.smooth_fbo.fbo_id, data, range_scale, smooth_color, smooth_color)

        if self.render_data.is_active(key):
            render_pose: Pose = self.render_data.get_pose(key)
            v_c: AngleFeature = render_pose.angle_data
            PoseFeatureLayer.pose_feature_shader.use(self.render_fbo.fbo_id, v_c, range_scale, render_color, render_color)
            self.draw_joint_labels(self.render_fbo, render_pose.angle_data)

    @staticmethod
    def draw_joint_labels(fbo: Fbo, feature: AngleFeature) -> None:
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
