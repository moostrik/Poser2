"""What a depthai message holds, as the backend hands it on: a frame's packed NV12."""

from typing import cast

import depthai as dai
import numpy as np


def packed_nv12(frame: dai.ImgFrame) -> np.ndarray:
    """The frame's Y rows then its interleaved UV rows, without the chip's padding: `getData` holds
    each row at the stride and the UV below a padded plane. A new array, safe to keep."""
    width, height, stride = frame.getWidth(), frame.getHeight(), frame.getStride()
    data: np.ndarray = cast(np.ndarray, frame.getData())
    uv_start: int = stride * frame.getPlaneHeight()
    y: np.ndarray = data[:stride * height].reshape(height, stride)[:, :width]
    uv: np.ndarray = data[uv_start:uv_start + stride * (height // 2)].reshape(height // 2, stride)[:, :width]
    return np.concatenate((y, uv))
