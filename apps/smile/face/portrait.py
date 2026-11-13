"""Where smile's portrait is cut from the camera frame around a face: its center, size and the eye line's angle; the
upright box the overlay draws; how near that box comes to the image edge; and the crop itself, rotated so the eyes
are level (as ComfyUI's face crop did)."""

import cv2
import numpy as np

from modules.inference.face import Face


def crop_geometry(face: Face, portrait_zoom: float, portrait_offset: float, portrait_width: int, portrait_height: int) -> tuple[np.ndarray, float, float, float]:
    """Where the portrait is cropped from the image: centre (pixels), width, height and the angle of
    the eye line (radians).

    zoom: portrait width relative to the face height; offset: shift up, relative to the face height.
    """
    x0, y0, x1, y1 = face.bbox
    real_size: float = y1 - y0

    # eye line -> roll angle; order the eyes left to right in the image
    eye_a, eye_b = face.kps[0], face.kps[1]
    if eye_a[0] > eye_b[0]:
        eye_a, eye_b = eye_b, eye_a
    angle: float = float(np.arctan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))
    axis_y: np.ndarray = np.array([-np.sin(angle), np.cos(angle)])    # portrait y axis (down) in the image

    crop_width: float = real_size * portrait_zoom
    crop_height: float = crop_width * portrait_height / portrait_width
    center: np.ndarray = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5]) - axis_y * portrait_offset * real_size
    return center, crop_width, crop_height, angle


def portrait_box(face: Face, portrait_zoom: float, portrait_offset: float, portrait_width: int, portrait_height: int) -> tuple[float, float, float, float]:
    """The portrait crop as an upright box in the image (x0, y0, x1, y1, pixels): its centre and size,
    without its rotation. It reaches past the image edge as far as `crop_margin` allows (`crop_edge`), or
    for a rejected face."""
    center, crop_width, crop_height, _ = crop_geometry(face, portrait_zoom, portrait_offset, portrait_width, portrait_height)
    return (float(center[0] - crop_width * 0.5), float(center[1] - crop_height * 0.5),
            float(center[0] + crop_width * 0.5), float(center[1] + crop_height * 0.5))


def crop_edge(face: Face, portrait_zoom: float, portrait_offset: float, portrait_width: int, portrait_height: int,
              image_width: int, image_height: int) -> float:
    """Distance of the portrait crop's upright box (`portrait_box`: what the overlay draws) from the
    nearest image edge, relative to the crop width; negative when it reaches past. The rotated crop
    that `portrait_crop` cuts may reach further at a tilted head: its corners are then filled from
    the edge pixels."""
    x0, y0, x1, y1 = portrait_box(face, portrait_zoom, portrait_offset, portrait_width, portrait_height)
    return min(x0, y0, image_width - x1, image_height - y1) / max(x1 - x0, 1e-6)


def portrait_crop(img: np.ndarray, face: Face, portrait_zoom: float, portrait_offset: float, portrait_width: int, portrait_height: int) -> np.ndarray:
    """Crop a portrait around the face, rotated so the eyes are level (as ComfyUI's face crop did)."""
    center, crop_width, crop_height, angle = crop_geometry(face, portrait_zoom, portrait_offset, portrait_width, portrait_height)
    axis_x: np.ndarray = np.array([np.cos(angle), np.sin(angle)])     # portrait x axis in the image
    axis_y: np.ndarray = np.array([-np.sin(angle), np.cos(angle)])    # portrait y axis (down) in the image

    half_x: np.ndarray = axis_x * crop_width * 0.5
    half_y: np.ndarray = axis_y * crop_height * 0.5

    # map each portrait pixel to the image: origin at the top left corner, axes scaled to pixels
    scale: float = crop_width / portrait_width
    origin: np.ndarray = center - half_x - half_y
    matrix: np.ndarray = np.array([[axis_x[0] * scale, axis_y[0] * scale, origin[0]],
                                   [axis_x[1] * scale, axis_y[1] * scale, origin[1]]])
    # the crop may extend past the image edge: fill that part with the edge pixels instead of black
    return cv2.warpAffine(img, matrix, (portrait_width, portrait_height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
