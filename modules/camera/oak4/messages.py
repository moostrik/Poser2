"""What a depthai message holds, as the backend hands it on: a frame's packed NV12, its moment of capture and its
latency."""

import logging
import time

import depthai as dai
import numpy as np

logger = logging.getLogger(__name__)

LATENCY_SMOOTHING: float = 0.1      # how hard each frame pulls the latency reading

_future_capture_warned: bool = False     # a capture time after now is logged once per run


def packed_nv12(frame: dai.ImgFrame) -> np.ndarray:
    """The frame's Y rows then its interleaved UV rows, without the chip's padding: `getData` holds
    each row at the stride and the UV below a padded plane. A new array, safe to keep."""
    width, height, stride = frame.getWidth(), frame.getHeight(), frame.getStride()
    data: np.ndarray = frame.getData()
    uv_start: int = stride * frame.getPlaneHeight()
    y: np.ndarray = data[:stride * height].reshape(height, stride)[:, :width]
    uv: np.ndarray = data[uv_start:uv_start + stride * (height // 2)].reshape(height // 2, stride)[:, :width]
    return np.concatenate((y, uv))


def capture_time(message: dai.ImgFrame | dai.Tracklets | dai.ImgDetections) -> float:
    """The message's moment of capture, the end of the exposure, on `time.perf_counter()`: now less
    the message's age on depthai's clock, which holds the device's capture time on the host's. The
    two clocks need no common origin; on Windows they are one counter, 3 µs apart (OAK4.md). A
    capture time after now cannot be, so such a message is stamped with its arrival instead."""
    global _future_capture_warned
    now: float = time.perf_counter()
    latency: float = (dai.Clock.now() - message.getTimestamp()).total_seconds()
    if latency >= 0.0:
        return now - latency
    if not _future_capture_warned:
        _future_capture_warned = True
        logger.warning(f'a message was captured {-latency * 1000.0:.1f} ms in the future by depthai\'s clock: '
                       f'stamped with its arrival (logged once)')
    return now


def smoothed_latency(previous: float, message: dai.ImgFrame | dai.Tracklets) -> float:
    """The latency reading in ms after this message, capture to arrival, on depthai's clock because
    the message's timestamp is. A reading for the panel; the frame's time is `capture_time`."""
    latency: float = (dai.Clock.now() - message.getTimestamp()).total_seconds() * 1000.0
    return latency if previous == 0.0 else previous + LATENCY_SMOOTHING * (latency - previous)
