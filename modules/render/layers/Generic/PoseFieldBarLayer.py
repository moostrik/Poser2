
# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Text import draw_box_string, text_init

from modules.pose.features import PoseFeature
from modules.pose.Pose import Pose, PoseField, ScalarPoseField

from modules.render.renderers.PoseMeshUtils import POSE_COLOR_LEFT, POSE_COLOR_RIGHT

from modules.DataHub import DataHub, DataType

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.PoseFeature import PoseFeature as PoseFeatureShader

class PoseScalarBarLayer(LayerBase):
    pose_feature_shader = PoseFeatureShader()

    def __init__(self, track_id: int, data_hub: DataHub, data_type: DataType, feature_type: ScalarPoseField,
                min_color=(0.0, 0.5, 1.0), max_color=(1.0, 0.2, 0.0),
                draw_labels: bool = True, range_scale: float = 1.0) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._data_type: DataType = data_type
        self._fbo: Fbo = Fbo()
        self._p_pose: Pose | None = None
        self._labels: list[str] = []

        self.feature_type: ScalarPoseField = feature_type
        self.min_color: tuple[float, float, float] = min_color
        self.max_color: tuple[float, float, float] = max_color
        self.draw_labels: bool = draw_labels
        self.range_scale: float = range_scale

        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        if not PoseScalarBarLayer.pose_feature_shader.allocated:
            PoseScalarBarLayer.pose_feature_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        if PoseScalarBarLayer.pose_feature_shader.allocated:
            PoseScalarBarLayer.pose_feature_shader.deallocate()

    def draw(self, rect: Rect, draw_labels: bool = True) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        if draw_labels:
            self.draw_joint_labels(self._labels, rect)

    def update(self) -> None:
        # shader gets reset on hot reload, so we need to check if it's allocated
        if not PoseScalarBarLayer.pose_feature_shader.allocated:
            PoseScalarBarLayer.pose_feature_shader.allocate(monitor_file=True)

        key: int = self._track_id

        pose: Pose | None = self._data_hub.get_item(self._data_type, key)

        if pose is self._p_pose:
            return # no update needed

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        feature = pose.get_feature(PoseField[self.feature_type.name])
        if not isinstance(feature, PoseFeature):
            raise ValueError(f"PoseFeatureLayer expected feature of type PoseFeature, got {type(feature)}")

        PoseScalarBarLayer.pose_feature_shader.use(self._fbo.fbo_id, feature, self.range_scale, self.min_color, self.max_color)

        joint_enum_type = feature.__class__.feature_enum()
        num_joints: int = len(feature)
        self._labels = [joint_enum_type(i).name for i in range(num_joints)]

        # self._fbo.begin()
        # self.draw_joint_labels(self._labels, Rect(0, 0, self._fbo.width, self._fbo.height))
        # self._fbo.end()

    @staticmethod
    def draw_joint_labels(labels: list[str], draw_rect: Rect) -> None:
        """Draw joint names at the bottom of each bar."""
        num_labels: int = len(labels)
        step: float = draw_rect.width / num_labels

        # Alternate colors for readability
        colors: list[tuple[float, float, float, float]] = [
            (*POSE_COLOR_LEFT, 1.0),
            (*POSE_COLOR_RIGHT, 1.0)
        ]

        for i in range(num_labels):
            string: str = labels[i]
            x: int = int(draw_rect.x + (i + 0.1) * step)
            y: int = int(draw_rect.y + draw_rect.height * 0.5 - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3), True) # type: ignore

