# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.DataHub import DataHub, PoseDataHubTypes
from modules.gl.LayerBase import LayerBase, Rect
from modules.render.meshes.AllMeshRenderer import AllMeshRenderer

from modules.render.renderers import CamBBoxRenderer, CamDepthTrackRenderer, CamImageRenderer, CamMeshRenderer

# from modules.render.layers.Generic.CamPoseMeshLayer import CamPoseMeshLayer


class CamCompositeLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, data_type: PoseDataHubTypes, image_renderer: CamImageRenderer,
                 line_width: int = 2,
                 mesh_color: tuple[float, float, float, float] | None = None,
                 bbox_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self._cam_id: int = cam_id
        self._fbo: Fbo = Fbo()
        self._line_width: int = int(line_width)
        self._data_type: PoseDataHubTypes = data_type

        self._image_renderer: CamImageRenderer = image_renderer
        self._depth_track_renderer: CamDepthTrackRenderer = CamDepthTrackRenderer(cam_id, data)
        self._bbox_renderer: CamBBoxRenderer = CamBBoxRenderer(cam_id, data, data_type, line_width, bbox_color)
        self._mesh_renderer: CamMeshRenderer = CamMeshRenderer(cam_id, data, data_type, line_width, mesh_color)

    @property
    def data_type(self) -> PoseDataHubTypes:
        return self._data_type
    @data_type.setter
    def data_type(self, value: PoseDataHubTypes) -> None:
        self._data_type = value
        self._bbox_renderer.data_type = value
        self._mesh_renderer.data_type = value

    @property
    def line_width(self) -> int:
        return self._line_width
    @line_width.setter
    def line_width(self, value: int) -> None:
        self._line_width = value
        self._bbox_renderer.line_width = value
        self._mesh_renderer.line_width = value

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
        self._mesh_renderer.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._image_renderer.deallocate()
        self._depth_track_renderer.deallocate()
        self._bbox_renderer.deallocate()
        self._mesh_renderer.deallocate()


    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)


    def update(self) -> None:
        # Update all layers
        self._depth_track_renderer.update()
        self._bbox_renderer.update()
        self._mesh_renderer.update()

        # Composite them into the FBO
        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        full_rect = Rect(0, 0, self._fbo.width, self._fbo.height)

        self._fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._fbo.begin()
        self._image_renderer.draw(full_rect)
        self._depth_track_renderer.draw(full_rect)
        self._bbox_renderer.draw(full_rect)
        self._mesh_renderer.draw(full_rect)
        self._fbo.end()