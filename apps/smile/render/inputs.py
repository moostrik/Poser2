from threading import Lock

import numpy as np

from ..face import FaceObservations
from modules.gl import ChannelOrder, Image


class Faces:
    """The newest frame of the face detector: set from its thread, read on the main thread."""

    def __init__(self) -> None:
        self._mutex: Lock = Lock()
        self._frame: FaceObservations | None = None

    def set(self, frame: FaceObservations) -> None:
        with self._mutex:
            self._frame = frame

    def get(self) -> FaceObservations | None:
        with self._mutex:
            return self._frame


class Inputs:
    """Images for the render. set_image() may be called from any thread; update() uploads the
    newest of each on the main thread. All images are OpenCV BGR."""

    def __init__(self) -> None:
        self.camera: Image =            Image(ChannelOrder.BGR)    # live camera frames
        self.capture: Image =           Image(ChannelOrder.BGR)    # the captured face, display crop
        self.capture_depth: Image =     Image(ChannelOrder.BGR)    # depth of the capture, mask of the flash out
        self.diffusion: Image =         Image(ChannelOrder.BGR)    # generated previews, portrait and grin frames
        self.diffusion_depth: Image =   Image(ChannelOrder.BGR)    # depth of the last grin frame, mask of the bloom
        self.white: Image =             Image(ChannelOrder.BGR)    # what the flash fades to
        self.faces: Faces =             Faces()         # the faces of the newest camera frame

        self._all: list[Image] = [self.camera, self.capture, self.capture_depth,
                                  self.diffusion, self.diffusion_depth, self.white]
        self.white.set_image(np.full((8, 8, 3), 255, np.uint8))

    def update(self) -> None:
        for image in self._all:
            image.update()

    def deallocate(self) -> None:
        for image in self._all:
            image.deallocate()
