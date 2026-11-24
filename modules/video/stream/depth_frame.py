"""Depth as a video stream: 12-bit HEVC on the iGPU, millimeters in the Y plane in `DEPTH_STEP_MM` steps over a
neutral chroma plane. Design: VIDEO.md, Depth."""

import numpy as np

from ..hardware import ClipCodec, Hardware

DEPTH_CODEC: ClipCodec = ClipCodec.HEVC
DEPTH_HARDWARE: Hardware = Hardware.iGPU
DEPTH_PIX_FMT: str = 'p012le'
DEPTH_QUALITY: int = 10         # constant quality, `-q:v`
DEPTH_STEP_MM: int = 3
NO_DEPTH_MM: int = 100          # the codec leaves up to 12 mm where there was none; nothing real is this close
DEPTH_NEUTRAL_UV: int = 2048 << 4
MAX_DEPTH_COUNT: int = 4095


def depth_frame_bytes(width: int, height: int) -> int:
    """The bytes of one `p012le` frame: 16-bit Y, and 16-bit UV at half height."""
    return width * height * 3


def depth_to_p012(depth: np.ndarray) -> np.ndarray:
    """uint16 millimeters (H, W) as one `p012le` frame, (H × 3/2, W) uint16: the Y plane over the UV plane."""
    height, width = depth.shape
    frame: np.ndarray = np.empty((height * 3 // 2, width), np.uint16)
    luma: np.ndarray = frame[:height]
    np.floor_divide(depth, DEPTH_STEP_MM, out=luma)
    np.minimum(luma, MAX_DEPTH_COUNT, out=luma)
    np.left_shift(luma, 4, out=luma)
    frame[height:] = DEPTH_NEUTRAL_UV
    return frame


def p012_to_depth(raw: bytes | bytearray, width: int, height: int) -> np.ndarray:
    """One `p012le` frame as uint16 millimeters (H, W), 0 where there is no depth."""
    luma: np.ndarray = np.frombuffer(raw, np.uint16, width * height).reshape(height, width)
    depth: np.ndarray = (luma >> 4) * DEPTH_STEP_MM
    depth[depth < NO_DEPTH_MM] = 0
    return depth
