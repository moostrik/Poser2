import logging
import os
import random

import cv2

from modules.gl import Image

logger = logging.getLogger(__name__)

LOGO_FOLDER = 'logos'


class Inputs:
    """Images for the render. set_image() may be called from any thread; update() uploads the
    newest of each on the render thread. All images are OpenCV BGR."""

    def __init__(self) -> None:
        self.camera: Image =            Image('BGR')    # live camera frames
        self.capture: Image =           Image('BGR')    # the captured face, display crop
        self.capture_depth: Image =     Image('BGR')    # depth of the capture, mask of the flash out
        self.diffusion: Image =         Image('BGR')    # generated previews, portrait and grin frames
        self.diffusion_depth: Image =   Image('BGR')    # depth of the last grin frame, mask of the bloom
        self.logo: Image =              Image('BGR')    # what the flash fades to

        self._all: list[Image] = [self.camera, self.capture, self.capture_depth,
                                  self.diffusion, self.diffusion_depth, self.logo]
        self._logo_files: list[str] = [f for f in os.listdir(LOGO_FOLDER) if f.endswith('.jpg')]
        if not self._logo_files:
            logger.warning('No logos (.jpg) in %s', LOGO_FOLDER)

    def update(self) -> None:
        for image in self._all:
            image.update()

    def deallocate(self) -> None:
        for image in self._all:
            image.deallocate()

    def load_logo(self) -> None:
        """Pick a random logo for the next flash."""
        if not self._logo_files:
            return
        path: str = os.path.join(LOGO_FOLDER, random.choice(self._logo_files))
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning('Failed to load logo %s', path)
            return
        self.logo.set_image(image)
