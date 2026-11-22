"""What an open OAK reads from its factory calibration: its lens, and its IMU's orientation against the camera."""

import logging

import depthai as dai

logger = logging.getLogger(__name__)


def read_lens_calibration(device: dai.Device, socket: dai.CameraBoardSocket, size: tuple[int, int],
                          device_id: str = '') -> tuple[float, float, float] | None:
    """This unit's lens from its factory calibration: (focal px/rad, cx, cy) at `size`, None when the device carries
    none, which is not an error. Read to report the unit's deviation from the shared lens (`lens_report`); the warp
    is built from the shared lens."""
    try:
        calibration = device.readCalibration()
        matrix = calibration.getCameraIntrinsics(socket, size[0], size[1])
        model = calibration.getDistortionModel(socket)
        spec_fov: float = float(calibration.getFov(socket, useSpec=True))
        fx, fy = float(matrix[0][0]), float(matrix[1][1])
        cx, cy = float(matrix[0][2]), float(matrix[1][2])
        logger.debug(f'{device_id} calibration {socket.name}: model {model}, spec fov {spec_fov:.1f} deg, '
                     f'fx {fx:.1f} fy {fy:.1f} px/rad, center ({cx:.1f}, {cy:.1f}) at {size[0]}x{size[1]}')
        return fx, cx, cy
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read calibration: {exc}')
        return None


def imu_rotation_to_camera(device: dai.Device, socket: dai.CameraBoardSocket,
                           device_id: str = '') -> list[list[float]] | None:
    """The IMU-to-camera extrinsics for this socket, 4x4, None if the board does not relate them. The readout
    applies them, and the board's own turn (`MountReadout`'s `board_roll`) after them."""
    try:
        matrix = device.readCalibration().getImuToCameraExtrinsics(socket)
        rotation = [[round(value, 4) for value in row[:3]] for row in matrix[:3]]
        logger.debug(f'{device_id} IMU-to-camera rotation {rotation}')
        return matrix
    except Exception as exc:
        logger.debug(f'{device_id} no IMU-to-camera extrinsics ({exc}): the readout takes the IMU frame as the camera\'s')
        return None
