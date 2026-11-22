"""Finding an OAK-D."""

import logging

from depthai import Device

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
