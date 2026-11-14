"""Finding an OAK-D, and what an open one says about itself: its sensors and its lens."""

from depthai import Device

import logging
logger = logging.getLogger(__name__)


def get_device_list(verbose: bool = False) -> list[str]:
    """The ids of the OAK-Ds depthai sees now."""
    device_list: list[str] = []
    if verbose:
        logger.info('-- CAMERAS --------------------------------------------------')
    for device in Device.getAllAvailableDevices():
        device_list.append(device.getMxId())
        if verbose:
            logger.info(f"Camera: {device.getMxId()} {device.state}")
    if verbose:
        logger.info('-------------------------------------------------------------')
    return device_list


def log_connected_sensors(device: Device, device_id: str = '') -> None:
    """Log the sensor behind each socket of an open device: the OAK-D Pro W and OAK-1 W ship with lenses 32°
    apart, so the hardware says which variant it is."""
    try:
        for feature in device.getConnectedCameraFeatures():
            logger.debug(f'{device_id} sensor {feature.socket.name}: {feature.sensorName} '
                         f'{feature.width}x{feature.height}')
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read sensor features: {exc}')


def read_lens_calibration(device: Device, socket, mode_size: tuple[int, int],
                          device_id: str = '') -> tuple[float, float, float] | None:
    """This unit's lens from its factory calibration: (focal px/rad, cx, cy) at `mode_size`, None when the device
    carries none. Read to report the unit's deviation from the shared lens and confirm its sensor variant; the
    warp is built from the shared lens."""
    try:
        calibration = device.readCalibration()
        matrix = calibration.getCameraIntrinsics(socket, mode_size[0], mode_size[1])
        model = calibration.getDistortionModel(socket)
        spec_fov: float = float(calibration.getFov(socket, useSpec=True))
        fx, fy = float(matrix[0][0]), float(matrix[1][1])
        cx, cy = float(matrix[0][2]), float(matrix[1][2])
        logger.debug(f'{device_id} calibration {socket.name}: model {model}, spec fov {spec_fov:.1f} deg, '
                     f'fx {fx:.1f} fy {fy:.1f} px/rad, center ({cx:.1f}, {cy:.1f}) '
                     f'at {mode_size[0]}x{mode_size[1]}')
        return fx, cx, cy
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read calibration: {exc}')
        return None
