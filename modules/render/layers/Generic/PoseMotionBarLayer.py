""" Draws a scalar bar for pose motion features """

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl import Fbo, Texture, draw_box_string, text_init
from modules.pose.features import PoseFeatureType
from modules.pose.Frame import Frame, FrameField
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.shaders import PoseMotionBar as shader

from modules.utils.HotReloadMethods import HotReloadMethods


POSE_COLOR_LEFT:            tuple[float, float, float] = (1.0, 0.5, 0.0) # Orange
POSE_COLOR_RIGHT:           tuple[float, float, float] = (0.0, 1.0, 1.0) # Cyan

class PoseMotionBarLayer(LayerBase):

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, feature_type: FrameField,
                line_thickness: float = 1.0, line_smooth: float = 1.0, color=(1.0, 1.0, 1.0, 1.0)) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None
        self._labels: list[str] = []

        self.data_type: PoseDataHubTypes = data_type
        self.feature_type: FrameField = feature_type
        self.color: tuple[float, float, float, float] = color
        self.bg_alpha: float = 0.4
        self.line_thickness: float = line_thickness
        self.line_smooth: float = line_smooth
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

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        if self.draw_labels:
            self._label_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:

        key: int = self._track_id

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), key)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        feature = pose.get_feature(FrameField[self.feature_type.name])
        if not isinstance(feature, PoseFeatureType):
            raise ValueError(f"PoseFeatureLayer expected feature of type PoseFeature, got {type(feature)}")

        # print(feature.values)

        line_thickness = 1.0 / self._fbo.height * self.line_thickness
        line_smooth = 1.0 / self._fbo.height * self.line_smooth

        self._shader.use(self._fbo.fbo_id, feature, line_thickness, line_smooth,
                                                   self.color, (*POSE_COLOR_RIGHT, self.bg_alpha), (*POSE_COLOR_LEFT, self.bg_alpha))

        joint_enum_type = feature.__class__.enum()
        num_joints: int = len(feature)
        labels: list[str] = [joint_enum_type(i).name for i in range(num_joints)]
        if labels != self._labels:
            PoseMotionBarLayer.render_labels(self._label_fbo, labels)
        self._labels = [joint_enum_type(i).name for i in range(num_joints)]


    @staticmethod
    def render_labels(fbo: Fbo,labels: list[str]) -> None:
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

