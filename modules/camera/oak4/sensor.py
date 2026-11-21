"""The OAK4's sensors: the color sensor and the stereo pair, the frame sizes asked of them, their depth models and
the ranges of their controls."""

from enum import IntEnum
from typing import Callable

import depthai as dai

from modules.utils import Rect

from ..geometry import CameraResolution, resolve_resolution, mode_size, mirrored_roi, flipped_roi, turned_roi

# The color sensor (IMX586), and with `depth_model` the OV9282 stereo pair. Grey is the ISP's luma of the color
# sensor. A resolution label here is a delivered size; the Camera node picks the sensor mode from it.
SENSOR_SOCKET: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A
STEREO_SOCKETS: tuple[dai.CameraBoardSocket, dai.CameraBoardSocket] = (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)

# The width over the height of the frame the Camera node crops itself: 5:4 at the sensor mode's rows, so the depth model's 8:5 frame is used further across than by a square.
FRAME_ASPECT: float = 5.0 / 4.0

# The direct frame's width is a multiple of this: the iGPU's H.264 decoder hangs on a clip of any
# other width (900 hangs, 896 and 720 play), so a frame of any other width cannot be replayed.
FRAME_WIDTH_ALIGNMENT: int = 16

# The stereo frames the depth model is fed; a larger input gained nothing and dropped frames.
STEREO_SIZE: tuple[int, int] = (640, 400)


class DepthModel(IntEnum):
    """The Neural Depth model on the camera, by its size. The sizes that hold 60 fps beside the
    color frames, measured: 576x360 and up do not (46 fps, 139 ms)."""
    OFF = 0
    NEURAL_288 = 1      # 288x180: 26 to 28 ms, capture to PC
    NEURAL_384 = 2      # 384x240: 31 to 41 ms
    NEURAL_480 = 3      # 480x300: 42 to 44 ms, and the boxes 8 ms later


DEPTH_MODELS: dict[DepthModel, dai.DeviceModelZoo] = {
    DepthModel.NEURAL_288: dai.DeviceModelZoo.NEURAL_DEPTH_288X180,
    DepthModel.NEURAL_384: dai.DeviceModelZoo.NEURAL_DEPTH_384X240,
    DepthModel.NEURAL_480: dai.DeviceModelZoo.NEURAL_DEPTH_480X300,
}


def requested_size(color: bool, resolution: CameraResolution) -> tuple[int, int]:
    """The size asked of the sensor for this label, before the warp and the crop."""
    return mode_size(color, resolve_resolution(color, resolution))


def direct_size(resolution: CameraResolution) -> tuple[int, int]:
    """The frame the Camera node crops itself: `FRAME_ASPECT` at the sensor mode's rows, the width on
    `FRAME_WIDTH_ALIGNMENT`."""
    rows: int = requested_size(True, resolution)[1]
    return (round(rows * FRAME_ASPECT / FRAME_WIDTH_ALIGNMENT) * FRAME_WIDTH_ALIGNMENT, rows)


# The control ranges. Exposure's floor and ISO are where this sensor clamps a manual value, measured
# on the IMX586 of an OAK-4-PRO-W at P720 and 60 fps; exposure's top and the rest are the ranges in
# the `CameraControl` docs. Auto exposure goes up to 16277 µs at that rate.
EXPOSURE_RANGE:     tuple[int, int] = (  86, 33000)
ISO_RANGE:          tuple[int, int] = (  65, 16585)
BALANCE_RANGE:      tuple[int, int] = (1000, 12000)
CONTRAST_RANGE:     tuple[int, int] = ( -10, 10   )
BRIGHTNESS_RANGE:   tuple[int, int] = ( -10, 10   )
LUMA_DENOISE_RANGE: tuple[int, int] = (   0, 4    )
CHROMA_DENOISE_RANGE: tuple[int, int] = ( 0, 4    )
SATURATION_RANGE:   tuple[int, int] = ( -10, 10   )
SHARPNESS_RANGE:    tuple[int, int] = (   0, 4    )
COMPENSATION_RANGE: tuple[int, int] = (  -9, 9    )
DEPTH_SCORE_RANGE:  tuple[int, int] = (   0, 255  )     # confidence, edge and the temporal delta
FAR_LIMIT_M:        float           = 12.0              # the far slider's top, the recording's full scale: no limit
NO_FAR_LIMIT_MM:    int             = 65535


class AntiBanding(IntEnum):
    """The mains flicker auto exposure avoids, by exposing in multiples of its half period."""
    OFF = 0
    MAINS_50_HZ = 1
    MAINS_60_HZ = 2
    AUTO = 3


ANTI_BANDING_MODES: dict[AntiBanding, dai.CameraControl.AntiBandingMode] = {
    AntiBanding.OFF: dai.CameraControl.AntiBandingMode.OFF,
    AntiBanding.MAINS_50_HZ: dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
    AntiBanding.MAINS_60_HZ: dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
    AntiBanding.AUTO: dai.CameraControl.AntiBandingMode.AUTO,
}


def sensor_region(region: Rect, rotate90: bool, flip_v: bool, flip_h: bool,
                  to_source: Callable[[float, float], tuple[float, float]]) -> tuple[int, int, int, int]:
    """A region of the picture as shown, in fractions of it, as the sensor pixels `setAutoExposureRegion`
    takes: (x, y, width, height). `to_source` maps a frame fraction to the sensor's pixels; the picture is
    turned and flipped after the camera as its frame's flags say, so the region is taken back through them first,
    in reverse order."""
    frame: Rect = mirrored_roi(region) if flip_h else region
    frame = flipped_roi(frame) if flip_v else frame
    frame = turned_roi(frame) if rotate90 else frame
    x0, y0 = to_source(frame.x, frame.y)
    x1, y1 = to_source(frame.x + frame.width, frame.y + frame.height)
    return (round(x0), round(y0), round(x1 - x0), round(y1 - y0))
