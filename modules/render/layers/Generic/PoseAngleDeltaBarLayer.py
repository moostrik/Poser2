
# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Text import draw_box_string, text_init

from modules.pose.features import PoseFeatureType
from modules.pose.Frame import Frame, FrameField

from modules.render.layers.meshes.PoseMeshUtils import POSE_COLOR_LEFT, POSE_COLOR_RIGHT

from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.PoseAngleDeltaBar import PoseAngleDeltaBar

class PoseAngleDeltaBarLayer(LayerBase):
    pose_feature_shader = PoseAngleDeltaBar()

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0, color=(1.0, 1.0, 1.0, 1.0)) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None
        self._labels: list[str] = []

        self.data_type: PoseDataHubTypes = data_type
        self.color: tuple[float, float, float, float] = color
        self.line_thickness: float = line_thickness
        self.line_smooth: float = line_smooth
        self.draw_labels: bool = False
        self.bg_alpha: float = 1.0

        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)


    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)

        if not PoseAngleDeltaBarLayer.pose_feature_shader.allocated:
            PoseAngleDeltaBarLayer.pose_feature_shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        if PoseAngleDeltaBarLayer.pose_feature_shader.allocated:
            PoseAngleDeltaBarLayer.pose_feature_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        if self.draw_labels:
            self._label_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # shader gets reset on hot reload, so we need to check if it's allocated
        if not PoseAngleDeltaBarLayer.pose_feature_shader.allocated:
            PoseAngleDeltaBarLayer.pose_feature_shader.allocate()

        key: int = self._track_id

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), key)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        line_thickness = 1.0 / self._fbo.height * self.line_thickness
        line_smooth = 1.0 / self._fbo.height * self.line_smooth

        PoseAngleDeltaBarLayer.pose_feature_shader.use(self._fbo.fbo_id, pose.angles, pose.angle_vel,
                                                       line_thickness, line_smooth,
                                                       (*POSE_COLOR_RIGHT, self.bg_alpha), (*POSE_COLOR_LEFT, self.bg_alpha))

        joint_enum_type = pose.angles.__class__.enum()
        num_joints: int = len(pose.angles)
        labels: list[str] = [joint_enum_type(i).name for i in range(num_joints)]
        if labels != self._labels:
            PoseAngleDeltaBarLayer.render_labels(self._label_fbo, labels)
        self._labels = [joint_enum_type(i).name for i in range(num_joints)]


    @staticmethod
    def render_labels(fbo: Fbo,labels: list[str]) -> None:
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
