"""Finding an OAK4, the link it is on, and what an open one says about itself: its sensors and its lens."""

import logging
from enum import IntEnum

import depthai as dai

from .. import CameraResolution

logger = logging.getLogger(__name__)

SYSTEM_RATE_HZ: float = 1.0     # how often the camera reports its CPU, memory and temperature


def get_device_list(verbose: bool = False) -> list[str]:
    """The ids of the OAK4s depthai sees now, each once: an OAK4 on both USB and Ethernet is listed
    by depthai on both. depthai 3 also sees the OAK-D cameras (RVC2) on the same PC; they are left
    out, so this backend never opens one."""
    device_list: list[str] = []
    if verbose:
        logger.info('-- CAMERAS --------------------------------------------------')
    for device in dai.Device.getAllAvailableDevices():
        is_oak4: bool = device.platform == dai.XLinkPlatform.X_LINK_RVC4
        if is_oak4 and device.getDeviceId() not in device_list:
            device_list.append(device.getDeviceId())
        if verbose:
            skipped: str = '' if is_oak4 else ' (not an OAK4: skipped)'
            logger.info(f"Camera: {device.getDeviceId()} {device.name} {device.platform} {device.protocol} {device.state}{skipped}")
    if verbose:
        logger.info('-------------------------------------------------------------')
    return device_list


class CameraLink(IntEnum):
    """How the camera reaches the PC. USB is the faster: measured 4 to 9 ms less capture to arrival."""
    USB = 0
    ETHERNET = 1


LINK_PROTOCOLS: dict[CameraLink, dai.XLinkProtocol] = {
    CameraLink.USB: dai.XLinkProtocol.X_LINK_USB_EP,
    CameraLink.ETHERNET: dai.XLinkProtocol.X_LINK_TCP_IP,
}


def link_name(protocol: dai.XLinkProtocol) -> str:
    """The `CameraLink` a route is, as the readings show it; depthai's own name for any other route."""
    return next((link.name for link, route in LINK_PROTOCOLS.items() if route == protocol), protocol.name)


def find_device(device_id: str, link: CameraLink) -> dai.DeviceInfo | None:
    """This OAK4 on `link`, or on the link it does have with a warning; None when it is on neither.
    Chosen here because depthai's own pick by id is whichever it finds first."""
    routes: list[dai.DeviceInfo] = [device for device in dai.Device.getAllAvailableDevices() if device.getDeviceId() == device_id]
    for route in routes:
        if route.protocol == LINK_PROTOCOLS[link]:
            return route
    if routes:
        logger.warning(f'{device_id} is not on {link.name}: using {routes[0].protocol}')
        return routes[0]
    return None


def link_resolution(protocol: dai.XLinkProtocol, resolution: CameraResolution, ethernet_resolution: CameraResolution) -> CameraResolution:
    """The frame the link the camera opened on can carry: Ethernet has less room than USB."""
    return ethernet_resolution if protocol == LINK_PROTOCOLS[CameraLink.ETHERNET] else resolution


def log_connected_sensors(device: dai.Device, device_id: str = '') -> None:
    """Log the sensor behind each socket of an open device, and the chip it runs on."""
    try:
        logger.info(f'{device_id} {device.getDeviceName()} on {device.getPlatformAsString()}')
        for feature in device.getConnectedCameraFeatures():
            logger.debug(f'{device_id} sensor {feature.socket.name}: {feature.sensorName} '
                         f'{feature.width}x{feature.height}')
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read sensor features: {exc}')


def read_lens_calibration(device: dai.Device, socket: dai.CameraBoardSocket, size: tuple[int, int],
                          device_id: str = '') -> tuple[float, float, float] | None:
    """This unit's lens from its factory calibration: (focal px/rad, cx, cy) at `size`, or None
    when the device carries none, which is not an error. Read so each unit's deviation from the
    settings file's shared lens can be reported (`lens_deviation`)."""
    try:
        calibration = device.readCalibration()
        matrix = calibration.getCameraIntrinsics(socket, size[0], size[1])
        model = calibration.getDistortionModel(socket)
        spec_fov: float = float(calibration.getFov(socket, True))
        fx, fy = float(matrix[0][0]), float(matrix[1][1])
        cx, cy = float(matrix[0][2]), float(matrix[1][2])
        logger.debug(f'{device_id} calibration {socket.name}: model {model}, spec fov {spec_fov:.1f} deg, '
                     f'fx {fx:.1f} fy {fy:.1f} px/rad, center ({cx:.1f}, {cy:.1f}) at {size[0]}x{size[1]}')
        return fx, cx, cy
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read calibration: {exc}')
        return None
