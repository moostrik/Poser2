""" Draws the full camera image, depth tracklets, pose lines, and bounding boxes."""

# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture, Blit
from modules.DataHub import DataHub, PoseDataHubTypes
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.render.layers.composite.BBoxCamRenderer import BBoxCamRenderer
from modules.render.layers.composite.TrackletCamRenderer import TrackletCamRenderer
from modules.render.layers.data.PoseLineLayer import PoseLineLayer

from modules.utils.HotReloadMethods import HotReloadMethods


class CamCompositeLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, data_type: PoseDataHubTypes, cam_texture: Texture,
                 line_width: float = 1.0) -> None:
        self._cam_id: int = cam_id
        self._fbo: Fbo = Fbo()
        self._line_width: float = line_width
        self._data_type: PoseDataHubTypes = data_type

        self._cam_texture: Texture = cam_texture
        self._depth_track_renderer: TrackletCamRenderer = TrackletCamRenderer(cam_id, data)
        self._bbox_renderer: BBoxCamRenderer = BBoxCamRenderer(cam_id, data, data_type, int(line_width), (1.0, 1.0, 1.0, 1.0))
        # Pose Points Layer works on track id, not cam id -> fix
        self._pose_line_layer: PoseLineLayer = PoseLineLayer(cam_id, data, data_type, line_width, 0.0, False, True, None)

        self.hot_reload = HotReloadMethods(self.__class__)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    @property
    def data_type(self) -> PoseDataHubTypes:
        return self._data_type
    @data_type.setter
    def data_type(self, value: PoseDataHubTypes) -> None:
        self._data_type = value
        self._bbox_renderer.data_type = value

    @property
    def line_width(self) -> float:
        return self._line_width
    @line_width.setter
    def line_width(self, value: float) -> None:
        self._line_width = value
        self._pose_line_layer.line_width = value
        self._bbox_renderer.line_width = int(value)

    @property
    def bbox_color(self) -> tuple[float, float, float, float]:
        return self._bbox_renderer.bbox_color
    @bbox_color.setter
    def bbox_color(self, value: tuple[float, float, float, float]) -> None:
        self._bbox_renderer.bbox_color = value

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._depth_track_renderer.allocate()
        self._bbox_renderer.allocate()
        self._pose_line_layer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._depth_track_renderer.deallocate()
        self._bbox_renderer.deallocate()
        self._pose_line_layer.deallocate()

    def update(self) -> None:
        # Update all layers
        self._depth_track_renderer.update()
        self._bbox_renderer.update()
        self._pose_line_layer.update()
        self._bbox_renderer.bbox_color = (1.0, 1.0, 1.0, 1.0)  # Example: set bbox color to red

        glEnable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        full_rect = Rect(0, 0, self._fbo.width, self._fbo.height)

        self._fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._fbo.begin()
        Blit.use(self._cam_texture)
        self._depth_track_renderer.draw(full_rect)
        self._bbox_renderer.draw(full_rect)
        self._pose_line_layer.draw(full_rect)
        self._fbo.end()