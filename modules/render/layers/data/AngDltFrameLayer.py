
# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl import Fbo, Texture, Blit, clear_color, draw_box_string, text_init
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import PoseAngleDeltaBar
from .Colors import POSE_COLOR_LEFT, POSE_COLOR_RIGHT, POSE_COLOR_CENTER

from modules.utils.HotReloadMethods import HotReloadMethods


class AngDltFrameLayer(LayerBase):
    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0, color=(1.0, 1.0, 1.0, 1.0)) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()
        self._labels: list[str] = []
        self._shader: PoseAngleDeltaBar = PoseAngleDeltaBar()

        self.data_type: PoseDataHubTypes = data_type
        self.color: tuple[float, float, float, float] = color
        self.line_thickness: float = line_thickness
        self.line_smooth: float = line_smooth
        self.draw_labels: bool = False
        self.bg_alpha: float = 1.0

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

    def draw(self) -> None:
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)
            if self.draw_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        line_thickness = 1.0 / self._fbo.height * self.line_thickness
        line_smooth = 1.0 / self._fbo.height * self.line_smooth

        self._fbo.begin()
        clear_color()
        self._shader.use(pose.angles, pose.angle_vel, line_thickness, line_smooth, (*POSE_COLOR_RIGHT, self.bg_alpha), (*POSE_COLOR_LEFT, self.bg_alpha))
        self._fbo.end()

        joint_enum_type = pose.angles.__class__.enum()
        num_joints: int = len(pose.angles)
        labels: list[str] = [joint_enum_type(i).name for i in range(num_joints)]
        if labels != self._labels:
            AngDltFrameLayer.render_labels(self._label_fbo, labels)
        self._labels = [joint_enum_type(i).name for i in range(num_joints)]


    @staticmethod
    def render_labels(fbo: Fbo,labels: list[str]) -> None:
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

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
