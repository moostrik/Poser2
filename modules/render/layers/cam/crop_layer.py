""" Draws camera image roi for a given pose frame """

# Local application imports
from modules.settings import Field, BaseSettings
from modules.board import HasFrames
from modules.gl import Fbo, Texture, clear_color
from modules.utils import Rect
from modules.pose import Frame, BBox
from ..layer_base import LayerBase, DataCache
from ...shaders import DrawRoi


class CropLayerSettings(BaseSettings):
    stage: Field[int] = Field(0, description="Pipeline stage for pose data")


class CropLayer(LayerBase):
    def __init__(self, track_id: int, board: HasFrames, cam_texture: Texture, settings: CropLayerSettings | None = None) -> None:
        self.settings: CropLayerSettings = settings or CropLayerSettings()
        self._track_id: int = track_id
        self._board: HasFrames = board
        self._fbo: Fbo = Fbo()
        self._cam_texture: Texture = cam_texture
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        # Add DrawRoi shader
        self._roi_shader = DrawRoi()

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
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._track_id)
        self._data_cache.set(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        pose_rect: Rect = pose[BBox].to_rect()
        # convert to texture space
        pose_rect = pose_rect.flip_y(0.5)

        self._fbo.begin()
        clear_color()
        self._roi_shader.use(self._cam_texture, pose_rect)
        self._fbo.end()