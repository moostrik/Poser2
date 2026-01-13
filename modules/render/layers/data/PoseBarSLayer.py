""" Draws lines representing pose motion similarity and magnitude """

# Third-party imports
from OpenGL.GL import * # type: ignore

import numpy as np

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl import Fbo, Texture, draw_box_string, text_init

from modules.pose.features import AggregationMethod
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.shaders import PoseValuesBar as shader

from modules.utils.HotReloadMethods import HotReloadMethods


POSE_COLOR_LEFT:            tuple[float, float, float] = (1.0, 0.5, 0.0) # Orange
POSE_COLOR_RIGHT:           tuple[float, float, float] = (0.0, 1.0, 1.0) # Cyan

class PoseBarSLayer(LayerBase):

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None
        self._labels: list[str] = []

        self.data_type: PoseDataHubTypes = data_type
        self.draw_labels: bool = True

        self._shader: shader = shader()
        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._label_fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._track_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

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
        # motion: float = np.nansum(pose.angle_motion.values)
        motion_array = np.array([motion], dtype=np.float32)
        motion_colors = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        motion_sim = similarity * motion
        motion_sim_colors = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [0.0, 1.0, 0.0, 1.0],  # Green
        ], dtype=np.float32)


        # print(motion_array)

        line_thickness = 1.0 / self._fbo.height * 1.0
        line_smooth = 1.0 / self._fbo.height * 4.0

        self._fbo.begin()
        self._shader.use(similarity, sim_colors, line_thickness, line_smooth * 10)
        self._shader.use(motion_array, motion_colors, line_thickness, line_smooth)
        self._shader.use(motion_sim, motion_sim_colors, line_thickness, line_smooth)
        self._fbo.end()

        return


        joint_enum_type = feature.__class__.enum()
        num_joints: int = len(feature)
        labels: list[str] = [joint_enum_type(i).name for i in range(num_joints)]
        if labels != self._labels:
            PoseBarSLayer.render_labels(self._label_fbo, labels)
        self._labels = [joint_enum_type(i).name for i in range(num_joints)]


    @staticmethod
    def render_labels(fbo: Fbo, labels: list[str]) -> None:
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

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
