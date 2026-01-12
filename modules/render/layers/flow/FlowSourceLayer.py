""" Optical Flow Layer - computes and visualizes optical flow from camera images """

# Standard library imports

# Third-party imports
import numpy as np
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub
from modules.gl import Texture, Image
from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase


class FlowSourceLayer(LayerBase):
    def __init__(self, cam_id: int, data_hub: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub

        # Image textures for uploading numpy arrays
        self.curr_image: Image = Image(channel_order='BGR')  # OpenCV uses BGR
        self.prev_image: Image = Image(channel_order='BGR')
        self._p_images: tuple[np.ndarray, np.ndarray] | None = None
        self._dirty: bool = False
        self._available: bool = False

    @property
    def texture(self) -> Texture:
        return self.curr_image

    @property
    def curr_texture(self) -> Texture:
        return self.texture

    @property
    def prev_texture(self) -> Texture:
        return self.prev_image

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def available(self) -> bool:
        return self._available


    def deallocate(self) -> None:
        self.prev_image.deallocate()
        self.curr_image.deallocate()

    def update(self) -> None:
        self._dirty = False

        images: tuple[np.ndarray, np.ndarray] | None = self._data_hub.get_item(DataHubType.flow_images, self._cam_id)

        if images is self._p_images:
            return
        self._p_images = images

        if images is None:
            self._available = False
            return
        self._available = True

        # Upload images to GPU textures
        self.prev_image.set_image(images[0])
        self.curr_image.set_image(images[1])
        self.prev_image.update()
        self.curr_image.update()

        self._dirty = True

