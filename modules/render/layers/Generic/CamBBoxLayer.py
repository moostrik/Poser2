""" Draws camera image roi for a given pose frame """

# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub
from modules.gl import Fbo, Texture
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.layers.source import ImageSourceLayer
from modules.render.shaders import DrawRoi

from modules.utils.HotReloadMethods import HotReloadMethods


class CamBBoxLayer(LayerBase):
    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, cam_texture: Texture) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._cam_texture: Texture = cam_texture
        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type

        # Add DrawRoi shader
        self._roi_shader = DrawRoi()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._roi_shader.deallocate()

    def update(self) -> None:
        key: int = self._cam_id

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), key)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        pose_rect = pose.bbox.to_rect()

        self._roi_shader.use(self._fbo.fbo_id, self._cam_texture.tex_id, pose_rect)