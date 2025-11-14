# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.DataHub import DataHub, DataType
from modules.gl.LayerBase import LayerBase, Rect
from modules.render.meshes.PoseMesh import PoseMesh

from modules.render.layers.Generic.CamImageLayer import CamImageLayer
from modules.render.layers.Generic.CamDepthTrackLayer import CamDepthTrackLayer
from modules.render.layers.Generic.CamPoseMeshLayer import CamPoseMeshLayer


class CamCompositeLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub, type: DataType, pose_meshes: PoseMesh,
                 bbox_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self._cam_id: int = cam_id
        self._fbo: Fbo = Fbo()

        self._image_layer: CamImageLayer = CamImageLayer(cam_id, data)
        self._depth_track_layer: CamDepthTrackLayer = CamDepthTrackLayer(cam_id, data)
        self._pose_layer: CamPoseMeshLayer = CamPoseMeshLayer(cam_id, data, type, pose_meshes, bbox_color)


    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._image_layer.allocate(width, height, internal_format)
        self._depth_track_layer.allocate(width, height, internal_format)
        self._pose_layer.allocate(width, height, internal_format)


    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._image_layer.deallocate()
        self._depth_track_layer.deallocate()
        self._pose_layer.deallocate()


    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)


    def update(self) -> None:
        # Update all layers
        self._image_layer.update()
        self._depth_track_layer.update()
        self._pose_layer.update()

        # Composite them into the FBO
        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._fbo.begin()

        full_rect = Rect(0, 0, self._fbo.width, self._fbo.height)
        self._image_layer.draw(full_rect)
        self._depth_track_layer.draw(full_rect)
        self._pose_layer.draw(full_rect)

        self._fbo.end()