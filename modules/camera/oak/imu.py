"""The OAK-D's IMU: its rate, its smoothing, and its turn on the board against the camera."""

from depthai import Device

import logging
logger = logging.getLogger(__name__)

IMU_RATE_HZ: int = 5
IMU_SMOOTHING: float = 0.1      # how hard each sample pulls the average: the camera is on a tripod, a reading is mostly noise
IMU_BOARD_ROLL: float = -90.0   # the IMU's turn on the board about the optical axis, measured (CAMERA.md, the mount readout)


def imu_rotation_to_camera(device: Device, socket, device_id: str = '') -> list[list[float]] | None:
    """The IMU-to-camera matrix for this socket, None if the board does not relate them; logged, since it decides
    whether the readout runs on the board's figures or on `IMU_BOARD_ROLL`."""
    try:
        matrix = device.readCalibration().getImuToCameraExtrinsics(socket)
        rotation = [[round(value, 4) for value in row[:3]] for row in matrix[:3]]
        logger.debug(f'{device_id} IMU-to-camera rotation {rotation}')
        return matrix
    except Exception as exc:
        logger.debug(f'{device_id} no IMU-to-camera extrinsics ({exc}) — '
                     f'falling back to the board constant, roll offset {IMU_BOARD_ROLL:.0f} deg')
        return None
