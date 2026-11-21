"""The view: the part of a camera frame the screen shows. It covers the output size with the frame's middle, scaled to
fill it without distortion; `view_rect` is that part as a rectangle of the frame, for the render to draw and the faces
to be measured against, and `cover` is the same fit as a resize, for an image kept on the host."""

import cv2
import numpy as np

from modules.utils import Rect


def view_rect(frame_width: int, frame_height: int, out_width: int, out_height: int) -> Rect:
    """The part of a frame_width x frame_height frame that covers out_width x out_height, its middle kept: 0..1 of
    the frame, the origin top-left."""
    scale: float = max(out_width / frame_width, out_height / frame_height)
    width: float = min(out_width / scale / frame_width, 1.0)
    height: float = min(out_height / scale / frame_height, 1.0)
    return Rect((1.0 - width) / 2.0, (1.0 - height) / 2.0, width, height)


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize the image to cover width x height and crop the middle: no distortion. An image of that
    size is returned as it is."""
    image_height, image_width = image.shape[:2]
    if (image_width, image_height) == (width, height):
        return image
    scale: float = max(width / image_width, height / image_height)
    resized: np.ndarray = cv2.resize(image, (max(width, round(image_width * scale)), max(height, round(image_height * scale))), interpolation=cv2.INTER_AREA)
    x0: int = (resized.shape[1] - width) // 2
    y0: int = (resized.shape[0] - height) // 2
    return np.ascontiguousarray(resized[y0:y0 + height, x0:x0 + width])
