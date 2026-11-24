"""Finding an OAK4, the link it is on, and the chip an open one runs on."""

import logging
from enum import IntEnum

import depthai as dai

from .. import CameraResolution

logger = logging.getLogger(__name__)


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


def log_platform(device: dai.Device, device_id: str = '') -> None:
    """Log the device's name and the chip it runs on (depthai 3 only: depthai 2 has no `getPlatformAsString`)."""
    try:
        logger.info(f'{device_id} {device.getDeviceName()} on {device.getPlatformAsString()}')
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read its platform: {exc}')
