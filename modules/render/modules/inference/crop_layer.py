""" Draws camera image roi for a given pose frame """

# Local application imports
from modules.board import HasFrames
from modules.gl import Fbo, Texture, clear_color
from modules.utils import Rect
from modules.pose import Frame, BBox
from ...layer_base import LayerBase
from ...data_cache import DataCache


class CropLayer(LayerBase):
    """The camera image cut to a track's box. `stage` is an input its owner sets: the pipeline stage of the box."""

    def __init__(self, track_id: int, board: HasFrames, cam_texture: Texture, stage: int = 0) -> None:
        self.stage: int = stage
        self._track_id: int = track_id
        self._board: HasFrames = board
        self._fbo: Fbo = Fbo()
        self._cam_texture: Texture = cam_texture
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()

    def update(self) -> None:
        pose: Frame | None = self._board.get_frame(self.stage, self._track_id)
        self._data_cache.set(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        pose_rect: Rect = pose[BBox].to_rect()

        self._fbo.begin()
        clear_color()
        self._cam_texture.draw(src=pose_rect)
        self._fbo.end()