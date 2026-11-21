"""The OAK4's IMU: its turn on the board against the camera."""

import logging

import depthai as dai

logger = logging.getLogger(__name__)

# The turn the board mounts the IMU at about the optical axis: a half turn on the OAK4, measured (an upright OAK4
# Pro W read a roll of -178.6 with this at 0); the OAK-D's is a quarter turn. A board constant, not a unit's
# `roll_offset`. Since depthai 3.6.1 `ACCELEROMETER_RAW` is the IMU's own frame, the one
# `getImuToCameraExtrinsics` describes.
IMU_BOARD_ROLL: float = 180.0


def imu_rotation_to_camera(device: dai.Device, socket: dai.CameraBoardSocket, device_id: str = '') -> list[list[float]] | None:
    """The IMU-to-camera matrix for this socket, or None if the board does not relate them;
    logged, since it decides whether the readout runs on the board's figures or on `IMU_BOARD_ROLL`."""
    try:
        matrix = device.readCalibration().getImuToCameraExtrinsics(socket)
        rotation = [[round(value, 4) for value in row[:3]] for row in matrix[:3]]
        logger.debug(f'{device_id} IMU-to-camera rotation {rotation}')
        return matrix
    except Exception as exc:
        logger.debug(f'{device_id} no IMU-to-camera extrinsics ({exc}): the board constant, '
                     f'roll offset {IMU_BOARD_ROLL:.0f} deg')
        return None
