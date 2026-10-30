"""Depth as a video stream: 12-bit HEVC on the iGPU, the only hardware path past 8 bits, so recording
depth costs the installation nothing. Millimetres go into the Y plane in `DEPTH_STEP_MM` steps,
12 m full scale, over a neutral chroma plane. It is lossy but close: on a synthetic hard-edged
frame the largest error was 24 mm and frame-to-frame noise grew from 8.6 to 9.0 mm. Do not use
`global_quality` here: at low values it breaks down to errors in metres; constant quality does not."""

import numpy as np

DEPTH_ENCODER: str = 'hevc_qsv'
DEPTH_PIX_FMT: str = 'p012le'
DEPTH_QUALITY: int = 10         # constant quality, `-q:v`
DEPTH_STEP_MM: int = 3
NO_DEPTH_MM: int = 100          # the codec leaves up to 12 mm where there was none; nothing real is this close
DEPTH_NEUTRAL_UV: int = 2048 << 4
MAX_DEPTH_COUNT: int = 4095


def depth_frame_bytes(width: int, height: int) -> int:
    """The bytes of one `p012le` frame: 16-bit Y, and 16-bit UV at half height."""
    return width * height * 3


def depth_to_p012(depth: np.ndarray) -> bytes:
    """uint16 millimetres (H, W) as one `p012le` frame."""
    luma: np.ndarray = np.minimum(depth // DEPTH_STEP_MM, MAX_DEPTH_COUNT).astype(np.uint16) << 4
    chroma: np.ndarray = np.full((depth.shape[0] // 2, depth.shape[1]), DEPTH_NEUTRAL_UV, np.uint16)
    return luma.tobytes() + chroma.tobytes()


def p012_to_depth(raw: bytes, width: int, height: int) -> np.ndarray:
    """One `p012le` frame as uint16 millimetres (H, W), 0 where there is no depth."""
    luma: np.ndarray = np.frombuffer(raw, np.uint16, width * height).reshape(height, width)
    depth: np.ndarray = (luma >> 4) * DEPTH_STEP_MM
    depth[depth < NO_DEPTH_MM] = 0
    return depth
