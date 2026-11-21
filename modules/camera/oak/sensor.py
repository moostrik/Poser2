"""The OAK-D's sensors: the depthai mode behind each resolution label, and the ranges of their controls."""

from depthai import MonoCameraProperties, ColorCameraProperties

from ..geometry import CameraResolution, resolve_resolution

# The depthai 2.30 modes behind the labels; which sensor has which label is `MONO_RESOLUTIONS`, `COLOR_RESOLUTIONS`.
MONO_MODES: dict[CameraResolution, MonoCameraProperties.SensorResolution] = {
    CameraResolution.P720: MonoCameraProperties.SensorResolution.THE_720_P,
    CameraResolution.P800: MonoCameraProperties.SensorResolution.THE_800_P,
}

COLOR_MODES: dict[CameraResolution, ColorCameraProperties.SensorResolution] = {
    CameraResolution.P720:  ColorCameraProperties.SensorResolution.THE_720_P,
    CameraResolution.P800:  ColorCameraProperties.SensorResolution.THE_800_P,
    CameraResolution.P1080: ColorCameraProperties.SensorResolution.THE_1080_P,
    CameraResolution.P2160: ColorCameraProperties.SensorResolution.THE_4_K,
}


def mono_mode(resolution: CameraResolution) -> MonoCameraProperties.SensorResolution:
    """The depthai mono sensor mode for this label."""
    return MONO_MODES[resolve_resolution(False, resolution)]


def color_mode(resolution: CameraResolution) -> ColorCameraProperties.SensorResolution:
    """The depthai color sensor mode for this label."""
    return COLOR_MODES[resolve_resolution(True, resolution)]


EXPOSURE_RANGE:     tuple[int, int] = (1000, 33000)
ISO_RANGE:          tuple[int, int] = ( 100, 1600 )
BALANCE_RANGE:      tuple[int, int] = (1000, 12000)
CONTRAST_RANGE:     tuple[int, int] = ( -10, 10   )
BRIGHTNESS_RANGE:   tuple[int, int] = ( -10, 10   )
LUMA_DENOISE_RANGE: tuple[int, int] = (   0, 4    )
SATURATION_RANGE:   tuple[int, int] = ( -10, 10   )
SHARPNESS_RANGE:    tuple[int, int] = (   0, 4    )
