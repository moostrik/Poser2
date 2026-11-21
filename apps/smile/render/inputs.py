"""The render's textures of the board's images, made on the main thread: the camera's full uploaded frame (on the GPU
already, `render.camera_image_layer`) and the CPU images, each uploaded once when the board holds a new one."""

import numpy as np

from modules.gl import ChannelOrder, Image
from modules.render import BoardImageLayer, camera_image_layer

from ..board import RenderBoard


class Inputs:
    """The board's images as textures, newest first, all OpenCV BGR but the camera's. `update` takes what is new."""

    def __init__(self, board: RenderBoard) -> None:
        self._board: RenderBoard = board
        self.camera: BoardImageLayer = camera_image_layer(0, board)     # the camera's full frame, upright
        self.capture: Image =           Image(ChannelOrder.BGR)    # the captured face, display crop
        self.capture_depth: Image =     Image(ChannelOrder.BGR)    # depth of the capture, mask of the flash out
        self.diffusion: Image =         Image(ChannelOrder.BGR)    # generated previews, portrait and grin frames
        self.diffusion_depth: Image =   Image(ChannelOrder.BGR)    # depth of the last grin frame, mask of the bloom
        self.white: Image =             Image(ChannelOrder.BGR)    # what the flash fades to

        self._images: list[Image] = [self.capture, self.capture_depth, self.diffusion, self.diffusion_depth, self.white]
        self._taken: dict[int, np.ndarray | None] = {}      # per texture, the board's image it shows
        self.white.set_image(np.full((8, 8, 3), 255, np.uint8))

    def update(self) -> None:
        self.camera.update()
        board: RenderBoard = self._board
        self._take(self.capture, board.get_capture())
        self._take(self.capture_depth, board.get_capture_depth())
        self._take(self.diffusion, board.get_diffusion_image())
        self._take(self.diffusion_depth, board.get_diffusion_depth())
        for image in self._images:
            image.update()

    def _take(self, image: Image, new: np.ndarray | None) -> None:
        """The board's image to the texture when it is a new one."""
        if new is not None and new is not self._taken.get(id(image)):
            self._taken[id(image)] = new
            image.set_image(new)

    def deallocate(self) -> None:
        self.camera.deallocate()
        for image in self._images:
            image.deallocate()
