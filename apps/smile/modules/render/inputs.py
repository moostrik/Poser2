import numpy as np

from modules.gl import Image


class Inputs:
    """Images for the render. set_image() may be called from any thread; update() uploads the
    newest of each on the render thread. All images are OpenCV BGR."""

    def __init__(self) -> None:
        self.camera: Image =            Image('BGR')    # live camera frames
        self.capture: Image =           Image('BGR')    # the captured face, display crop
        self.capture_depth: Image =     Image('BGR')    # depth of the capture, mask of the flash out
        self.diffusion: Image =         Image('BGR')    # generated previews, portrait and grin frames
        self.diffusion_depth: Image =   Image('BGR')    # depth of the last grin frame, mask of the bloom
        self.white: Image =             Image('BGR')    # what the flash fades to

        self._all: list[Image] = [self.camera, self.capture, self.capture_depth,
                                  self.diffusion, self.diffusion_depth, self.white]
        self.white.set_image(np.full((8, 8, 3), 255, np.uint8))

    def update(self) -> None:
        for image in self._all:
            image.update()

    def deallocate(self) -> None:
        for image in self._all:
            image.deallocate()
