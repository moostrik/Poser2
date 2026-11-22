"""The sensors of an open OAK, as it lists them."""

import logging

import depthai as dai

logger = logging.getLogger(__name__)


def log_connected_sensors(device: dai.Device, device_id: str = '') -> None:
    """Log the sensor behind each socket of an open device: the hardware says which variant it is (the OAK-D Pro W
    and OAK-1 W ship with lenses 32° apart)."""
    try:
        for feature in device.getConnectedCameraFeatures():
            logger.debug(f'{device_id} sensor {feature.socket.name}: {feature.sensorName} '
                         f'{feature.width}x{feature.height}')
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read sensor features: {exc}')
