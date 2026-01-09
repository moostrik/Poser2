""" Draws the full camera image, depth tracklets, pose lines, and bounding boxes."""

# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture
from modules.DataHub import DataHub, PoseDataHubTypes
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.render.layers.renderers import CamBBoxRenderer, CamDepthTrackRenderer, CamImageRenderer
from modules.render.layers.generic.PoseLineLayer import PoseLineLayer

from modules.utils.HotReloadMethods import HotReloadMethods


class CamCompositeLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, data_type: PoseDataHubTypes, image_renderer: CamImageRenderer,
                 line_width: float = 1.0) -> None:
        self._cam_id: int = cam_id
        self._fbo: Fbo = Fbo()
        self._line_width: float = line_width
        self._data_type: PoseDataHubTypes = data_type

        self._image_renderer: CamImageRenderer = image_renderer
        self._depth_track_renderer: CamDepthTrackRenderer = CamDepthTrackRenderer(cam_id, data)
        self._pose_points_layer: PoseLineLayer = PoseLineLayer(cam_id, data, data_type, line_width, 0.0, False, True, None)
        self._bbox_renderer: CamBBoxRenderer = CamBBoxRenderer(cam_id, data, data_type, int(line_width), (1.0, 1.0, 1.0, 1.0))

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
        self._pose_points_layer.line_width = value
        self._bbox_renderer.line_width = int(value)

    @property
    def bbox_color(self) -> tuple[float, float, float, float]:
        return self._bbox_renderer.bbox_color
    @bbox_color.setter
    def bbox_color(self, value: tuple[float, float, float, float]) -> None:
        self._bbox_renderer.bbox_color = value

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._image_renderer.allocate()
        self._depth_track_renderer.allocate()
        self._bbox_renderer.allocate()
        self._pose_points_layer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._image_renderer.deallocate()
        self._depth_track_renderer.deallocate()
        self._bbox_renderer.deallocate()
        self._pose_points_layer.deallocate()

    def update(self) -> None:
        # Update all layers
        self._depth_track_renderer.update()
        self._bbox_renderer.update()
        self._pose_points_layer.update()
        self._bbox_renderer.bbox_color = (1.0, 1.0, 1.0, 1.0)  # Example: set bbox color to red

        glEnable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        full_rect = Rect(0, 0, self._fbo.width, self._fbo.height)

        self._fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._fbo.begin()
        self._image_renderer.draw(full_rect)
        self._depth_track_renderer.draw(full_rect)
        self._bbox_renderer.draw(full_rect)
        self._pose_points_layer.draw(full_rect)
        self._fbo.end()