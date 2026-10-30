"""What the oak4 backend knows about depthai 3 and the OAK4: the frame types, the control ranges,
the board's IMU, and the reads off an open device. The frame geometry every camera shares is
`modules.camera.geometry`; this maps it onto the device."""

import logging
from enum import IntEnum
from typing import Callable

import depthai as dai
import numpy as np

from .. import CameraResolution, resolve_resolution, mode_size, TrackingStatus
from modules.utils import Rect

logger = logging.getLogger(__name__)


# This backend delivers the color sensor (IMX586), and with `depth_model` depth off the OV9282 stereo
# pair on CAM_B and CAM_C. Grey comes off the color sensor as the ISP's luma. The contract's
# resolution labels are delivered sizes here, not sensor modes: the Camera node picks the sensor mode
# from the size requested (`geometry.mode_size`).
SENSOR_SOCKET: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A
STEREO_SOCKETS: tuple[dai.CameraBoardSocket, dai.CameraBoardSocket] = (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)

# The width over the height of the frame the Camera node crops itself: 5:4 at the sensor mode's rows, so the depth model's 8:5 frame is used further across than by a square.
FRAME_ASPECT: float = 5.0 / 4.0

# The direct frame's width is a multiple of this: the iGPU's H.264 decoder hangs on a clip of any
# other width (900 hangs, 896 and 720 play), so a frame of any other width cannot be replayed.
FRAME_WIDTH_ALIGNMENT: int = 16

# The stereo frames the depth model is fed; a larger input gained nothing and dropped frames.
STEREO_SIZE: tuple[int, int] = (640, 400)


class DepthModel(IntEnum):
    """The Neural Depth model on the camera, by its size. The sizes that hold 60 fps beside the
    color frames, measured: 576x360 and up do not (46 fps, 139 ms)."""
    OFF = 0
    NEURAL_288 = 1      # 288x180: 26 to 28 ms, capture to PC
    NEURAL_384 = 2      # 384x240: 31 to 41 ms
    NEURAL_480 = 3      # 480x300: 42 to 44 ms, and the boxes 8 ms later


DEPTH_MODELS: dict[DepthModel, tuple[dai.DeviceModelZoo, int]] = {  # the model and its rows
    DepthModel.NEURAL_288: (dai.DeviceModelZoo.NEURAL_DEPTH_288X180, 180),
    DepthModel.NEURAL_384: (dai.DeviceModelZoo.NEURAL_DEPTH_384X240, 240),
    DepthModel.NEURAL_480: (dai.DeviceModelZoo.NEURAL_DEPTH_480X300, 300),
}


def depth_size(model: DepthModel, frame: tuple[int, int]) -> tuple[int, int]:
    """The aligned depth's size for a delivered `frame`: the model's rows, which is all the detail
    it has, and an even width at the frame's aspect."""
    rows: int = DEPTH_MODELS[model][1]
    return (round(rows * frame[0] / frame[1] / 2.0) * 2, rows)


def requested_size(color: bool, resolution: CameraResolution) -> tuple[int, int]:
    """The size asked of the sensor for this label, before the warp and the crop."""
    return mode_size(color, resolve_resolution(color, resolution))


def direct_size(resolution: CameraResolution) -> tuple[int, int]:
    """The frame the Camera node crops itself: `FRAME_ASPECT` at the sensor mode's rows, the width on
    `FRAME_WIDTH_ALIGNMENT`."""
    rows: int = requested_size(True, resolution)[1]
    return (round(rows * FRAME_ASPECT / FRAME_WIDTH_ALIGNMENT) * FRAME_WIDTH_ALIGNMENT, rows)


# The control ranges. Exposure's floor and ISO are where this sensor clamps a manual value, measured
# on the IMX586 of an OAK-4-PRO-W at P720 and 60 fps; exposure's top and the rest are the ranges in
# the `CameraControl` docs. Auto exposure goes up to 16277 µs at that rate.
EXPOSURE_RANGE:     tuple[int, int] = (  86, 33000)
ISO_RANGE:          tuple[int, int] = (  65, 16585)
BALANCE_RANGE:      tuple[int, int] = (1000, 12000)
CONTRAST_RANGE:     tuple[int, int] = ( -10, 10   )
BRIGHTNESS_RANGE:   tuple[int, int] = ( -10, 10   )
LUMA_DENOISE_RANGE: tuple[int, int] = (   0, 4    )
CHROMA_DENOISE_RANGE: tuple[int, int] = ( 0, 4    )
SATURATION_RANGE:   tuple[int, int] = ( -10, 10   )
SHARPNESS_RANGE:    tuple[int, int] = (   0, 4    )
COMPENSATION_RANGE: tuple[int, int] = (  -9, 9    )
DEPTH_SCORE_RANGE:  tuple[int, int] = (   0, 255  )     # confidence, edge and the temporal delta
FAR_LIMIT_M:        float           = 12.0              # the far slider's top, the recording's full scale: no limit
NO_FAR_LIMIT_MM:    int             = 65535


class AntiBanding(IntEnum):
    """The mains flicker auto exposure avoids, by exposing in multiples of its half period."""
    OFF = 0
    MAINS_50_HZ = 1
    MAINS_60_HZ = 2
    AUTO = 3


ANTI_BANDING_MODES: dict[AntiBanding, dai.CameraControl.AntiBandingMode] = {
    AntiBanding.OFF: dai.CameraControl.AntiBandingMode.OFF,
    AntiBanding.MAINS_50_HZ: dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
    AntiBanding.MAINS_60_HZ: dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
    AntiBanding.AUTO: dai.CameraControl.AntiBandingMode.AUTO,
}


def sensor_region(region: Rect, flip_h: bool, to_source: Callable[[float, float], tuple[float, float]]) -> tuple[int, int, int, int]:
    """A region of the picture as shown, in fractions of it, as the sensor pixels `setAutoExposureRegion`
    takes: (x, y, width, height). `to_source` maps a frame fraction to the sensor's pixels; the picture
    is mirrored after the camera when `flip_h`, so the region is mirrored back first."""
    frame: Rect = mirrored_roi(region) if flip_h else region
    x0, y0 = to_source(frame.x, frame.y)
    x1, y1 = to_source(frame.x + frame.width, frame.y + frame.height)
    return (round(x0), round(y0), round(x1 - x0), round(y1 - y0))


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


def packed_nv12(frame: dai.ImgFrame) -> np.ndarray:
    """The frame's Y rows then its interleaved UV rows, without the chip's padding: `getData` holds
    each row at the stride and the UV below a padded plane. A new array, safe to keep."""
    width, height, stride = frame.getWidth(), frame.getHeight(), frame.getStride()
    data: np.ndarray = frame.getData()
    uv_start: int = stride * frame.getPlaneHeight()
    y: np.ndarray = data[:stride * height].reshape(height, stride)[:, :width]
    uv: np.ndarray = data[uv_start:uv_start + stride * (height // 2)].reshape(height // 2, stride)[:, :width]
    return np.concatenate((y, uv))


LATENCY_SMOOTHING: float = 0.1      # how hard each frame pulls the latency reading


def smoothed_latency(previous: float, message: dai.ImgFrame | dai.Tracklets) -> float:
    """The latency reading in ms after this message, capture to arrival, on depthai's clock because
    the message's timestamp is. A reading for the panel; nothing downstream takes time from it."""
    latency: float = (dai.Clock.now() - message.getTimestamp()).total_seconds() * 1000.0
    return latency if previous == 0.0 else previous + LATENCY_SMOOTHING * (latency - previous)


def mirrored_roi(roi: Rect) -> Rect:
    """A normalised box in the frame flipped left to right."""
    return Rect(1.0 - roi.x - roi.width, roi.y, roi.width, roi.height)


def mirrored_keypoints(keypoints: np.ndarray | None) -> np.ndarray | None:
    """Keypoints (N, 3) with x flipped left to right. The labels stay: a left shoulder stays the
    model's left shoulder, as the model saw the frame unflipped."""
    if keypoints is None:
        return None
    mirrored: np.ndarray = keypoints.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


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
SYSTEM_RATE_HZ: float = 1.0     # how often the camera reports its CPU, memory and temperature

# How hard each new sample pulls the running average: the camera is on a tripod, a single
# reading is almost entirely noise and there is no motion to track.
IMU_SMOOTHING: float = 0.1

# The turn the board mounts the IMU at about the optical axis. On the OAK-D it is a measured
# quarter turn (`oak.definitions.IMU_BOARD_ROLL`); on the OAK4 a half turn, measured: an upright
# OAK4 Pro W read a roll of -178.6 with this at 0. It is a board constant and not in `roll_offset`,
# which is per unit. Since depthai 3.6.1 `ACCELEROMETER_RAW` is the IMU's own frame, unrotated,
# which is the frame `getImuToCameraExtrinsics` describes.
IMU_BOARD_ROLL: float = 180.0


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


# ---------------------------------------------------------------------------
#  The pose model and the tracker
# ---------------------------------------------------------------------------
# The OAK4's chip takes a model compiled by Luxonis, fetched once from their zoo as an NN archive
# (`python -m modules.camera.oak4.models`, local files in `data/models`, not in git). Two chains give
# the 17 body landmarks (COCO, the order of `modules.pose.PointLandmark`), `ModelChain` picks:
# ONE_MODEL finds the people and their landmarks in one pass, decoded by `DetectionNetwork` from the
# archive's config; TWO_MODELS runs a detector, cuts one crop per person on the device and runs a
# top-down pose model on each crop, through Luxonis's depthai-nodes. DETECTOR gives boxes only, for
# an app that poses the crops on the host: yolov8n at 640x512, the delivered frame's 5:4, compiled
# through Luxonis's HubAI, as the zoo has no such size for this chip.


class ModelChain(IntEnum):
    OFF = 0             # no model on the camera: frames only
    DETECTOR = 1        # YOLOv8n: boxes only
    ONE_MODEL = 2       # YOLOv8-pose: boxes and keypoints in one pass
    TWO_MODELS = 3      # YOLOv6 detector, then Lite-HRNet on each person's crop


PERSON_DETECTOR_ARCHIVE: str = 'yolov8n_coco_640x512_rvc4.tar.xz'
POSE_MODEL: str = 'luxonis/yolov8-nano-pose-estimation:coco-512x288'
POSE_ARCHIVE: str = 'yolov8n_pose_coco_512x288_rvc4.tar.xz'
DETECTOR_MODEL: str = 'luxonis/yolov6-nano:r2-coco-512x288'
DETECTOR_ARCHIVE: str = 'yolov6n_coco_512x288_rvc4.tar.xz'
TOP_DOWN_MODEL: str = 'luxonis/lite-hrnet:18-coco-192x256'
TOP_DOWN_ARCHIVE: str = 'lite_hrnet18_coco_192x256_rvc4.tar.xz'
NUM_KEYPOINTS: int = 17
POSE_CROP_PADDING: float = 0.0          # margin round the person's box before the crop, normalised to the frame
POSE_CONFIDENCE_THRESHOLD: float = 0.5
POSE_OVERLAP_THRESHOLD: float = 0.5

# The device tracker's statuses, as the contract names them.
TRACKING_STATUS: dict[dai.Tracklet.TrackingStatus, TrackingStatus] = {
    dai.Tracklet.TrackingStatus.NEW: TrackingStatus.NEW,
    dai.Tracklet.TrackingStatus.TRACKED: TrackingStatus.TRACKED,
    dai.Tracklet.TrackingStatus.LOST: TrackingStatus.LOST,
    dai.Tracklet.TrackingStatus.REMOVED: TrackingStatus.REMOVED,
}
TRACKER_PERSON_LABEL: int = 0
TRACKER_TYPE: dai.TrackerType = dai.TrackerType.SHORT_TERM_IMAGELESS    # this chip has no ZERO_TERM_IMAGELESS, the `oak` backend's type


def model_archives(chain: ModelChain) -> tuple[str | None, str | None]:
    """The (pose, detector) archive names of a chain: `build_pipeline` reads one model off the one
    that is set, and the two-model chain off both."""
    if chain == ModelChain.DETECTOR:
        return None, PERSON_DETECTOR_ARCHIVE
    if chain == ModelChain.TWO_MODELS:
        return TOP_DOWN_ARCHIVE, DETECTOR_ARCHIVE
    if chain == ModelChain.ONE_MODEL:
        return POSE_ARCHIVE, None
    return None, None


def nn_frame_type(archive: dai.NNArchive) -> dai.ImgFrame.Type:
    """The frame type the archive's model takes, off its config; BGR888i when it does not say."""
    try:
        name: str = str(archive.getConfig().model.inputs[0].preprocessing.daiType)
        return getattr(dai.ImgFrame.Type, name.split('.')[-1])
    except Exception as exc:
        logger.debug(f'archive names no frame type ({exc}): BGR888i')
        return dai.ImgFrame.Type.BGR888i


def tracklet_keypoints(detection: dai.ImgDetection) -> np.ndarray | None:
    """The detection's landmarks as the contract carries them: (17, 3) float32, x and y normalised
    to the frame the detector saw (the delivered frame, stretched into the model), and the score;
    None when the detection carries none."""
    points = detection.getKeypoints()
    if len(points) != NUM_KEYPOINTS:
        return None
    return np.array([(p.imageCoordinates.x, p.imageCoordinates.y, p.confidence) for p in points], dtype=np.float32)


def keypoints_in_frame(message: dai.beta.Keypoints, frame: dai.ImgTransformation | None) -> np.ndarray | None:
    """A crop's keypoints from the top-down pose model, in the delivered frame, as the contract
    carries them: (17, 3) float32, x and y normalised to the frame, and the score. The message's
    transformation is the crop's, the frame's is the delivered frame's (both descend from the same
    sensor, so depthai remaps one into the other); None without a transformation to remap through,
    or when the count is not 17."""
    points = message.getKeypoints()
    if len(points) != NUM_KEYPOINTS:
        return None
    source = message.getTransformation()
    if source is None or frame is None:
        return None
    width, height = frame.getSize()
    rows: list[tuple[float, float, float]] = []
    for p in points:
        c = p.imageCoordinates
        q = source.remapPointTo(frame, dai.Point2f(c.x, c.y, True))
        if q.isNormalized():
            rows.append((q.x, q.y, p.confidence))
        else:
            rows.append((q.x / width, q.y / height, p.confidence))
    return np.array(rows, dtype=np.float32)
