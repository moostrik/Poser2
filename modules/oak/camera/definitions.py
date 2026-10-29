# https://blobconverter.luxonis.com/

"""What the oak backend knows about depthai: sensor modes, the detector and tracker constants, the
control ranges, the device queues, and the board's IMU. The frame geometry every camera shares is
`modules.camera.geometry`; this maps it onto the device."""

from pathlib import Path

import math
from enum import Enum, IntEnum, auto
from depthai import (Tracklet, TrackerType, Device, OpenVINO,
                     SpatialLocationCalculatorAlgorithm, MonoCameraProperties, ColorCameraProperties)

from modules.camera import CameraResolution, resolve_resolution, TrackingStatus

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Sensor modes — the device's enum for each `CameraResolution` label
# ---------------------------------------------------------------------------
# The labels and which sensor has which are `modules.camera.geometry` (`MONO_RESOLUTIONS`,
# `COLOR_RESOLUTIONS`); these are the depthai 2.30 enums behind them.
MONO_MODES: dict[CameraResolution, MonoCameraProperties.SensorResolution] = {
    CameraResolution.P720: MonoCameraProperties.SensorResolution.THE_720_P,
    CameraResolution.P800: MonoCameraProperties.SensorResolution.THE_800_P,
}

COLOR_MODES: dict[CameraResolution, ColorCameraProperties.SensorResolution] = {
    CameraResolution.P720:  ColorCameraProperties.SensorResolution.THE_720_P,
    CameraResolution.P800:  ColorCameraProperties.SensorResolution.THE_800_P,
    CameraResolution.P1080: ColorCameraProperties.SensorResolution.THE_1080_P,
}


def mono_mode(resolution: CameraResolution) -> MonoCameraProperties.SensorResolution:
    """The depthai mono sensor mode for this label."""
    return MONO_MODES[resolve_resolution(False, resolution)]


def color_mode(resolution: CameraResolution) -> ColorCameraProperties.SensorResolution:
    """The depthai colour sensor mode for this label."""
    return COLOR_MODES[resolve_resolution(True, resolution)]


# The device tracker's statuses, as the contract names them. The device's `ObjectTracker` has
# exactly these four, so every tracklet it hands over maps.
TRACKING_STATUS: dict[Tracklet.TrackingStatus, TrackingStatus] = {
    Tracklet.TrackingStatus.NEW: TrackingStatus.NEW,
    Tracklet.TrackingStatus.TRACKED: TrackingStatus.TRACKED,
    Tracklet.TrackingStatus.LOST: TrackingStatus.LOST,
    Tracklet.TrackingStatus.REMOVED: TrackingStatus.REMOVED,
}


# The wide blobs are built on tools.luxonis.com from the stock ultralytics yolov8n weights,
# OpenVINO 2022.1, at 512 x 448 for the tall cylindrical frame (1280 x 1152 at P800 is 1.111;
# 512 x 448 is 1.143; the P720 frame at 1280 x 960 is 1.333, a 17 % mismatch that a 512 x 384
# blob would close). Same pixel count as the 640 x 352 they replace (229 k vs 225 k), so the
# same inference cost — confirmed on the rig, `tracker_fps` at the input rate on all four
# cameras. The 640 x 352 files stay in data/models for reference; nothing selects them.
YOLOV8_WIDE_5S: str = "yolov8n_coco_512x448_5S.blob"
YOLOV8_WIDE_6S: str = "yolov8n_coco_512x448_6S.blob"
YOLOV8_WIDE_7S: str = "yolov8n_coco_512x448_7S.blob"
YOLOV8_SQUARE_5S: str = "yolov8n_coco_416x416_5S.blob"
YOLOV8_SQUARE_6S: str = "yolov8n_coco_416x416_6S.blob"
YOLOV8_SQUARE_7S: str = "yolov8n_coco_416x416_7S.blob"
YOLO_CONFIDENCE_THRESHOLD: float = 0.5
YOLO_OVERLAP_THRESHOLD: float = 0.5


def detector_input_size(blob_path: str | Path) -> tuple[int, int]:
    """(width, height) the detector blob was compiled for, read from the blob itself.

    Not a camera dimension: the frame is resized into this whatever the sensor mode is, with
    `setKeepAspectRatio(False)` (see the Yolo setups in pipeline.py), so the frame is STRETCHED
    into the blob, never cropped and never letterboxed. Two consequences: nothing is lost at the
    frame edges, so nobody goes undetected there; and detections come back in normalized
    coordinates of the detector input, which a pure stretch preserves, so they map 1:1 onto the
    full frame. An aspect mismatch between frame and blob costs detection *quality*, never
    *geometry* — which is why the tracker's azimuths are right whatever blob is in use.

    Read off the blob rather than kept as a constant beside the file name, because the two used
    to be separate declarations that nothing checked against each other. depthai stores the
    input as [W, H, C, N].
    """
    blob = OpenVINO.Blob(Path(blob_path))
    dims = next(iter(blob.networkInputs.values())).dims
    return int(dims[0]), int(dims[1])

TRACKER_PERSON_LABEL: int = 0
TRACKER_TYPE: TrackerType = TrackerType.ZERO_TERM_IMAGELESS
# ZERO_TERM_COLOR_HISTOGRAM higher accuracy (but can drift when losing object)
# ZERO_TERM_IMAGELESS slightly faster

DEPTH_TRACKER_LOCATION: SpatialLocationCalculatorAlgorithm = SpatialLocationCalculatorAlgorithm.MIN
DEPTH_TRACKER_BOX_SCALE: float = 1.0
DEPTH_TRACKER_MIN_DEPTH: int = 500
DEPTH_TRACKER_MAX_DEPTH: int = 10000

EXPOSURE_RANGE:     tuple[int, int] = (1000, 33000)
ISO_RANGE:          tuple[int, int] = ( 100, 1600 )
BALANCE_RANGE:      tuple[int, int] = (1000, 12000)
CONTRAST_RANGE:     tuple[int, int] = ( -10, 10   )
BRIGHTNESS_RANGE:   tuple[int, int] = ( -10, 10   )
LUMA_DENOISE_RANGE: tuple[int, int] = (   0, 4    )
SATURATION_RANGE:   tuple[int, int] = ( -10, 10   )
SHARPNESS_RANGE:    tuple[int, int] = (   0, 4    )

STEREO_DEPTH_RANGE: tuple[int, int] = ( 500, 15000)
STEREO_BRIGHTNESS_RANGE: tuple[int, int] = (   0, 255  )

class StereoMedianFilterType(Enum):
    OFF = 0
    KERNEL_3x3 = 1
    KERNEL_5x5 = 2
    KERNEL_7x7 = 3

STEREO_FILTER_NAMES: list[str] = [e.name for e in StereoMedianFilterType]

class Input(IntEnum):
    COLOR_CONTROL = auto()
    MONO_CONTROL = auto()
    STEREO_CONTROL = auto()
    VIDEO_FRAME_IN = auto()
    LEFT_FRAME_IN = auto()
    RIGHT_FRAME_IN = auto()

class Output(IntEnum):
    VIDEO_FRAME_OUT = auto()
    LEFT_FRAME_OUT = auto()
    RIGHT_FRAME_OUT = auto()
    STEREO_FRAME_OUT = auto()
    SYNC_FRAMES_OUT = auto()
    TRACKLETS_OUT = auto()
    IMU_OUT = auto()

def get_device_list(verbose: bool = False) -> list[str]:
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
    """Log the sensor behind each socket of an already-open device.

    Which sensor a device carries decides its field of view, and the OAK-D Pro W and OAK-1 W
    both ship in two variants whose lenses differ by 32 degrees horizontally (see the sensor
    reference above). Reading it off the hardware beats inferring it from which resolutions a
    preset happens to request.
    """
    try:
        for feature in device.getConnectedCameraFeatures():
            logger.debug(f'{device_id} sensor {feature.socket.name}: {feature.sensorName} '
                         f'{feature.width}x{feature.height}')
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read sensor features: {exc}')


# ---------------------------------------------------------------------------
#  IMU — the board's own orientation sensor
# ---------------------------------------------------------------------------
# `orientation_from_gravity` (modules.camera.geometry) turns the gravity vector into tilt and
# roll; this is how the vector gets from the IMU's frame into the camera's.

IMU_RATE_HZ: int = 5

# How hard each new sample pulls the running average. The camera is bolted to a tripod, so a
# single reading is almost entirely noise and there is no motion to track.
IMU_SMOOTHING: float = 0.1

# The IMU does not share the camera's axes: it sits a quarter turn about the optical axis, and
# `getImuToCameraExtrinsics` does not encode that rotation.
#
# MEASURED, NOT ASSUMED. On the rig all four cameras reported a roll of -90 degrees while their
# tilt read correctly, and that combination has exactly one cause: a rotation about the optical
# axis (z) leaves `gz` untouched and `hypot(gx, gy)` invariant, so tilt survives it intact while
# roll is displaced by a constant. Any other misalignment would have corrupted the tilt too.
#
# It is a property of the board, not of an installation, so it is a constant here rather than a
# setting — four separately adjusted tripods cannot agree on a quarter turn by coincidence.
#
# THIS CONSTANT CLAIMS THE QUARTER TURN AND NOTHING MORE. Calibrating against a spirit level left
# a residual of about -0.85 degrees shared by three of four units, and it is tempting to fold that
# in and call the constant -90.85. It is not folded in: -90 is a board *layout*, exact and
# evidenced, whereas -0.85 is the mean of three samples with no evidence a different set of boards
# shares it. Everything below a degree is per-unit and belongs in `CameraReadings.roll_offset`,
# where it is a measurement of one specific camera rather than an extrapolation from three. The
# fourth unit on this rig needs -2.25 while sitting level — a sensor fault, probably from a fall —
# which no shared constant could ever have absorbed.
IMU_BOARD_ROLL: float = -90.0


def imu_to_camera(vector: tuple[float, float, float],
                  extrinsics: list[list[float]] | None) -> tuple[float, float, float]:
    """Rotate a vector from the IMU's frame into a camera's.

    ``extrinsics`` is the 4x4 from ``CalibrationHandler.getImuToCameraExtrinsics``; only its
    rotation block is used, since a direction has no position. A missing or malformed matrix
    falls through as the identity — on a board where the two frames happen to agree that is
    right, and where they do not it is at least not silently wrong in an invented direction.
    """
    if not extrinsics or len(extrinsics) < 3:
        return vector
    try:
        return tuple(                                       # type: ignore[return-value]
            sum(extrinsics[row][col] * vector[col] for col in range(3))
            for row in range(3)
        )
    except (IndexError, TypeError):
        return vector


def unroll_imu_frame(vector: tuple[float, float, float],
                     board_roll: float = IMU_BOARD_ROLL) -> tuple[float, float, float]:
    """Undo the IMU's mounting rotation about the optical axis.

    A rotation about z, so it corrects the roll and cannot disturb the tilt — which is the reason
    the tilt readout was already trustworthy before this existed. See ``IMU_BOARD_ROLL``.
    """
    # Under this matrix a rotation by `angle` maps a roll of phi to phi - angle, so the angle to
    # apply IS the offset being removed, not its negation.
    angle: float = math.radians(board_roll)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)


def read_lens_calibration(device: Device, socket, mode_size: tuple[int, int],
                          device_id: str = '') -> tuple[float, float, float] | None:
    """This unit's lens from its factory calibration: (focal px/rad, cx, cy) at `mode_size`,
    or None when the device carries none, which is not an error.

    The factory model is a pinhole with a rational distortion polynomial (`Perspective`, 14
    coefficients). Its `fx` is the scale at the optical axis, and on these lenses the fitted
    distortion keeps radius linear in angle to 0.5 % out to 75 degrees — so `fx` IS the
    equidistant focal, and `(cx, cy)` the optical centre, which is all `source_lens` needs.
    `getCameraIntrinsics` at the mode size already accounts for the 720-row centre crop (it
    shifts `cy` by 40).

    The warp is NOT built from this: the shared lens in the preset is. This is read so each
    unit's deviation from that shared lens can be reported (`lens_deviation`), and so the
    sensor variant is confirmed — the OAK-D Pro W ships with lenses 32 degrees apart, and
    `lens_field` of `fx` tells them apart (129 vs ~97).
    """
    try:
        calibration = device.readCalibration()
        matrix = calibration.getCameraIntrinsics(socket, mode_size[0], mode_size[1])
        model = calibration.getDistortionModel(socket)
        spec_fov: float = float(calibration.getFov(socket, useSpec=True))
        fx, fy = float(matrix[0][0]), float(matrix[1][1])
        cx, cy = float(matrix[0][2]), float(matrix[1][2])
        logger.debug(f'{device_id} calibration {socket.name}: model {model}, spec fov {spec_fov:.1f} deg, '
                     f'fx {fx:.1f} fy {fy:.1f} px/rad, centre ({cx:.1f}, {cy:.1f}) '
                     f'at {mode_size[0]}x{mode_size[1]}')
        return fx, cx, cy
    except Exception as exc:                        # never let diagnostics break an open
        logger.debug(f'{device_id} could not read calibration: {exc}')
        return None


def imu_rotation_to_camera(device: Device, socket, device_id: str = '') -> list[list[float]] | None:
    """The IMU-to-camera matrix for this socket, or None if the board does not relate them.

    Logged rather than swallowed: whether this matrix exists decides whether the readout is
    running on the board's own figures or on `IMU_BOARD_ROLL`, and that is worth knowing from the
    log instead of inferring it from a suspicious roll reading — which is how the quarter turn was
    found in the first place.
    """
    try:
        matrix = device.readCalibration().getImuToCameraExtrinsics(socket)
        rotation = [[round(value, 4) for value in row[:3]] for row in matrix[:3]]
        logger.debug(f'{device_id} IMU-to-camera rotation {rotation}')
        return matrix
    except Exception as exc:
        logger.debug(f'{device_id} no IMU-to-camera extrinsics ({exc}) — '
                     f'falling back to the board constant, roll offset {IMU_BOARD_ROLL:.0f} deg')
        return None
