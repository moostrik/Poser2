
# Third-party imports
from OpenGL.GL import * # type: ignore

import numpy as np

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Text import draw_box_string, text_init

from modules.pose.features import PoseFeatureType, AngleMotion, AggregationMethod, SingleValue
from modules.pose.Frame import Frame, FrameField

from modules.render.layers.meshes.PoseMeshUtils import POSE_COLOR_LEFT, POSE_COLOR_RIGHT

from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.ValuesBar import ValuesBar as shader

class PoseMotionSimLayer(LayerBase):
    _shader = shader()

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None
        self._labels: list[str] = []

        self.data_type: PoseDataHubTypes = data_type
        self.draw_labels: bool = True

        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)


    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)

        if not PoseMotionSimLayer._shader.allocated:
            PoseMotionSimLayer._shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        if PoseMotionSimLayer._shader.allocated:
            PoseMotionSimLayer._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # if self.draw_labels:
        #     self._label_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # shader gets reset on hot reload, so we need to check if it's allocated
        if not PoseMotionSimLayer._shader.allocated:
            PoseMotionSimLayer._shader.allocate(monitor_file=True)

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._track_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        similarity = pose.similarity.values
        sim_colors = np.array([
            [1.0, 0.0, 0.0, 0.5],  # Red
            [1.0, 1.0, 0.0, 0.5],  # Yellow
            [0.0, 1.0, 0.0, 0.5]   # Green
        ], dtype=np.float32)
        motion = pose.angle_motion.aggregate(AggregationMethod.MAX)
        motion_array = np.array([motion], dtype=np.float32)
        motion_colors = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        motion_sim = similarity * motion
        motion_sim_colors = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [0.0, 1.0, 0.0, 1.0],  # Green
        ], dtype=np.float32)


        # print(feature.values)

        line_thickness = 1.0 / self._fbo.height * 1.0
        line_smooth = 1.0 / self._fbo.height * 4.0

        PoseMotionSimLayer._shader.use(self._fbo.fbo_id, similarity, sim_colors, line_thickness, line_smooth * 10)
        PoseMotionSimLayer._shader.use(self._fbo.fbo_id, motion_array, motion_colors, line_thickness, line_smooth)
        PoseMotionSimLayer._shader.use(self._fbo.fbo_id, motion_sim, motion_sim_colors, line_thickness, line_smooth)
        # PoseMotionSimLayer._shader.use(self._fbo.fbo_id, motion_sim, line_thickness, line_smooth,
        #                                            color, (*POSE_COLOR_RIGHT, 1.0), (*POSE_COLOR_LEFT, 1.0))
        # PoseMotionSimLayer._shader.use(self._fbo.fbo_id, motion_feature, line_thickness, line_smooth,
        #                                            color, color, color)



        return


        joint_enum_type = feature.__class__.enum()
        num_joints: int = len(feature)
        labels: list[str] = [joint_enum_type(i).name for i in range(num_joints)]
        if labels != self._labels:
            PoseMotionSimLayer.render_labels(self._label_fbo, labels)
        self._labels = [joint_enum_type(i).name for i in range(num_joints)]


    @staticmethod
    def render_labels(fbo: Fbo, labels: list[str]) -> None:
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        LayerBase.setView(fbo.width, fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        fbo.clear(0.0, 0.0, 0.0, 0.0)

        fbo.begin()

        """Draw joint names at the bottom of each bar."""
        num_labels: int = len(labels)
        if num_labels == 0:
            return
        step: float = rect.width / num_labels

        # Alternate colors for readability
        colors: list[tuple[float, float, float, float]] = [
            (*POSE_COLOR_LEFT, 1.0),
            (*POSE_COLOR_RIGHT, 1.0)
        ]

        for i in range(num_labels):
            string: str = labels[i]
            x: int = int(rect.x + (i + 0.1) * step)
            y: int = int(rect.y + rect.height * 0.5 - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3)) # type: ignore

        fbo.end()
