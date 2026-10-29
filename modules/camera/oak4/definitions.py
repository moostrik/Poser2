"""What the oak4 backend knows about depthai 3 and the OAK4: the frame types, the control ranges,
the board's IMU, and the reads off an open device. The frame geometry every camera shares is
`modules.camera.geometry`; this maps it onto the device."""

import logging

import depthai as dai

from .. import CameraResolution, resolve_resolution, mode_size

logger = logging.getLogger(__name__)


# The OAK4 has one sensor, colour (IMX586 on the OAK4 S); grey comes off the same sensor as the
# ISP's luma. The contract's resolution labels are delivered sizes here, not sensor modes: the
# Camera node picks the sensor mode from the size requested (`geometry.mode_size`).
SENSOR_SOCKET: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A


def frame_type(color: bool) -> dai.ImgFrame.Type:
    """The frame type requested from the sensor: interleaved BGR, which `getCvFrame` hands over as
    it is, or 8-bit grey, the one type the chip's warp block takes whole."""
    return dai.ImgFrame.Type.BGR888i if color else dai.ImgFrame.Type.GRAY8


def requested_size(color: bool, resolution: CameraResolution) -> tuple[int, int]:
    """The size asked of the sensor for this label, before the warp and the crop."""
    return mode_size(color, resolve_resolution(color, resolution))


# The control ranges of the OAK-D, kept until the IMX586's own are read off the hardware: the
# `CameraControl` setters are the same calls on depthai 3.
EXPOSURE_RANGE:     tuple[int, int] = (1000, 33000)
ISO_RANGE:          tuple[int, int] = ( 100, 1600 )
BALANCE_RANGE:      tuple[int, int] = (1000, 12000)
CONTRAST_RANGE:     tuple[int, int] = ( -10, 10   )
BRIGHTNESS_RANGE:   tuple[int, int] = ( -10, 10   )
LUMA_DENOISE_RANGE: tuple[int, int] = (   0, 4    )
SATURATION_RANGE:   tuple[int, int] = ( -10, 10   )
SHARPNESS_RANGE:    tuple[int, int] = (   0, 4    )


def get_device_list(verbose: bool = False) -> list[str]:
    """The ids of the devices depthai sees now, USB and Ethernet alike."""
    device_list: list[str] = []
    if verbose:
        logger.info('-- CAMERAS --------------------------------------------------')
    for device in dai.Device.getAllAvailableDevices():
        device_list.append(device.getDeviceId())
        if verbose:
            logger.info(f"Camera: {device.getDeviceId()} {device.name} {device.platform} {device.state}")
    if verbose:
        logger.info('-------------------------------------------------------------')
    return device_list


def log_connected_sensors(device: dai.Device, device_id: str = '') -> None:
    """Log the sensor behind each socket of an open device, and the chip it runs on."""
    try:
        logger.info(f'{device_id} {device.getDeviceName()} on {device.getPlatformAsString()}')
        for feature in device.getConnectedCameraFeatures():
            logger.debug(f'{device_id} sensor {feature.socket.name}: {feature.sensorName} '
                         f'{feature.width}x{feature.height}')
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read sensor features: {exc}')


# ---------------------------------------------------------------------------
#  IMU — the board's own orientation sensor
# ---------------------------------------------------------------------------
# `orientation_from_gravity` (modules.camera.geometry) turns the gravity vector into tilt and
# roll, `imu_to_camera` and `unroll_imu_frame` get the vector there.

IMU_RATE_HZ: int = 5

# How hard each new sample pulls the running average: the camera is on a tripod, a single
# reading is almost entirely noise and there is no motion to track.
IMU_SMOOTHING: float = 0.1

# The turn the board mounts the IMU at about the optical axis. On the OAK-D it is a measured
# quarter turn (`oak.definitions.IMU_BOARD_ROLL`); on the OAK4 it is unmeasured, so this claims
# nothing until a camera standing level reads a roll, which goes here as a board constant and
# not in `roll_offset`, which is per unit. Since depthai 3.6.1 `ACCELEROMETER_RAW` is the IMU's
# own frame, unrotated, which is the frame `getImuToCameraExtrinsics` describes.
IMU_BOARD_ROLL: float = 0.0


def read_lens_calibration(device: dai.Device, socket: dai.CameraBoardSocket, size: tuple[int, int],
                          device_id: str = '') -> tuple[float, float, float] | None:
    """This unit's lens from its factory calibration: (focal px/rad, cx, cy) at `size`, or None
    when the device carries none, which is not an error. Read so each unit's deviation from the
    preset's shared lens can be reported (`lens_deviation`), and so the lens the unit ships with
    is known: `lens_field` of `fx` tells the 95 degree wide lens from the 66 degree standard one."""
    try:
        calibration = device.readCalibration()
        matrix = calibration.getCameraIntrinsics(socket, size[0], size[1])
        model = calibration.getDistortionModel(socket)
        spec_fov: float = float(calibration.getFov(socket, True))
        fx, fy = float(matrix[0][0]), float(matrix[1][1])
        cx, cy = float(matrix[0][2]), float(matrix[1][2])
        logger.debug(f'{device_id} calibration {socket.name}: model {model}, spec fov {spec_fov:.1f} deg, '
                     f'fx {fx:.1f} fy {fy:.1f} px/rad, centre ({cx:.1f}, {cy:.1f}) at {size[0]}x{size[1]}')
        return fx, cx, cy
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read calibration: {exc}')
        return None


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
