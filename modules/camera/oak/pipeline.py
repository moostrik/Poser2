"""The depthai pipeline of one OAK-D camera: its source (the sensor or clips), the warp, the video output, the
detector and tracker, and the IMU."""

from pathlib import Path
import logging
import math

import cv2
import depthai as dai
import numpy as np

from ..geometry import CameraResolution, IMU_RATE_HZ, MountConfig, mode_size, frame_size, WARP_MESH, warp_mesh_points, \
    source_lens

from .models import YOLOV8_WIDE_6S, YOLOV8_WIDE_7S, YOLOV8_SQUARE_6S, YOLOV8_SQUARE_7S, \
    DETECTOR_CONFIDENCE_THRESHOLD, DETECTOR_OVERLAP_THRESHOLD, detector_input_size, TRACKER_PERSON_LABEL, TRACKER_TYPE
from .sensor import mono_mode, color_mode

logger = logging.getLogger(__name__)


# The XLink streams between the device and the host, named once for both ends.
VIDEO_STREAM: str = 'video'             # the delivered frames
TRACKLETS_STREAM: str = 'tracklets'     # the object tracker's output
IMU_STREAM: str = 'imu'                 # the accelerometer, on a device
CONTROL_STREAM: str = 'control'         # the sensor controls, on a device
CLIPS_STREAM: str = 'clips'             # the clip camera's frames, in


def model_path(models: str, square: bool, clips: bool) -> Path:
    """The detector blob for this frame shape. The clip camera's blob has one shave more: with no
    sensor on the device, the ISP's share of the SHAVEs is free for the network."""
    if square:
        name: str = YOLOV8_SQUARE_7S if clips else YOLOV8_SQUARE_6S
    else:
        name = YOLOV8_WIDE_7S if clips else YOLOV8_WIDE_6S
    return (Path(models) / name).resolve().absolute()


def build_pipeline(
    pipeline: dai.Pipeline,
    models: str,
    fps: float,
    color: bool,
    square: bool,
    yolo: bool,
    resolution: CameraResolution,
    mount: MountConfig,
    frame_height: int = 0,
    clips: bool = False,
    warp_clips: bool = False,
) -> None:
    """Build the camera's pipeline into `pipeline`; `clips` takes the frames from the host, through the warp with
    `warp_clips`; `frame_height` is the delivered height (`delivered_height`), 0 the mode's rows."""
    src_size: tuple[int, int] = mode_size(color, resolution)
    out_size: tuple[int, int] = frame_size(color, resolution, square, frame_height)
    logger.info("OAK pipeline: %s, %s, %s%s%s", 'square' if square else 'wide', 'color' if color else 'mono',
                'yolo, ' if yolo else '', 'clips' if clips else 'sensor', ' through the warp' if clips and warp_clips else '')

    if clips:
        source: dai.Node.Output = add_clip_input(pipeline, src_size, out_size, color)
    else:
        source = add_sensor(pipeline, color, resolution, fps)
    if clips and not warp_clips:
        frames: dai.Node.Output = source
    else:
        frames = add_warp(pipeline, source, src_size, out_size, color, mount, clips)
    add_video_output(pipeline, frames)
    if yolo:
        add_detector(pipeline, frames, square, model_path(models, square, clips))
    if not clips:
        add_imu(pipeline)


def add_sensor(pipeline: dai.Pipeline, color: bool, resolution: CameraResolution, fps: float) -> dai.Node.Output:
    """The sensor at its mode, with its control stream: color off `preview` (planar BGR), mono off the left
    camera; returns its frame output."""
    camera: dai.node.ColorCamera | dai.node.MonoCamera
    if color:
        camera = pipeline.create(dai.node.ColorCamera)
        camera.setResolution(color_mode(resolution))
        camera.setFps(fps)
        camera.setInterleaved(False)
        camera.setPreviewSize(*mode_size(True, resolution))
        frames: dai.Node.Output = camera.preview
    else:
        camera = pipeline.create(dai.node.MonoCamera)
        camera.setCamera("left")
        camera.setResolution(mono_mode(resolution))
        camera.setFps(fps)
        frames = camera.out

    control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    control.setStreamName(CONTROL_STREAM)
    control.out.link(camera.inputControl)
    return frames


def add_clip_input(pipeline: dai.Pipeline, src_size: tuple[int, int], out_size: tuple[int, int],
                   color: bool) -> dai.Node.Output:
    """The clip camera's frames, in. The ceiling takes a clip at the sensor's size or the delivered one."""
    channels: int = 3 if color else 1
    clips: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    clips.setStreamName(CLIPS_STREAM)
    clips.setMaxDataSize(max(src_size[0] * src_size[1], out_size[0] * out_size[1]) * channels)
    return clips.out


def add_warp(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int], out_size: tuple[int, int],
             color: bool, mount: MountConfig, clips: bool) -> dai.Node.Output:
    """The warp from the sensor's frame to the delivered one; for clips, only the part a
    recording does not carry yet (`clip_warp_mesh`). Returns its output."""
    warp: dai.node.Warp = pipeline.create(dai.node.Warp)
    if clips:
        mesh, mesh_w, mesh_h = clip_warp_mesh(src_size, out_size, src_size[0], mount)
    else:
        mesh, mesh_w, mesh_h = build_warp_mesh(src_size, out_size, src_size[0], mount)
    warp.setMaxOutputFrameSize(out_size[0] * out_size[1] * (3 if color else 1))
    warp.setOutputSize(*out_size)
    warp.setWarpMesh(mesh, mesh_w, mesh_h)
    source.link(warp.inputImage)
    return warp.out


def add_video_output(pipeline: dai.Pipeline, frames: dai.Node.Output) -> None:
    video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
    video.setStreamName(VIDEO_STREAM)
    frames.link(video.input)


def add_detector(pipeline: dai.Pipeline, frames: dai.Node.Output, square: bool, blob: Path) -> None:
    """The detector branch: the frame stretched into the blob (a square one kept), YOLO, the device's object
    tracker on the network's passthrough, so only on inferred frames, and the tracklet output."""
    manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.initialConfig.setResize(*detector_input_size(blob))
    manip.initialConfig.setKeepAspectRatio(square)
    frames.link(manip.inputImage)

    network: dai.node.YoloDetectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    network.setBlobPath(blob)
    network.setNumInferenceThreads(2)
    network.setNumClasses(80)
    network.setCoordinateSize(4)
    network.setConfidenceThreshold(DETECTOR_CONFIDENCE_THRESHOLD)
    network.setIouThreshold(DETECTOR_OVERLAP_THRESHOLD)
    network.input.setBlocking(False)
    manip.out.link(network.input)

    tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
    tracker.setTrackerType(TRACKER_TYPE)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
    network.passthrough.link(tracker.inputTrackerFrame)
    network.passthrough.link(tracker.inputDetectionFrame)
    network.out.link(tracker.inputDetections)

    tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
    tracklets.setStreamName(TRACKLETS_STREAM)
    tracker.out.link(tracklets.input)


def add_imu(pipeline: dai.Pipeline) -> None:
    """The raw accelerometer at `IMU_RATE_HZ`, for the mount readout; a board without an IMU never sends."""
    imu: dai.node.IMU = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, IMU_RATE_HZ)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    imu_out: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
    imu_out.setStreamName(IMU_STREAM)
    imu.out.link(imu_out.input)


# ---------------------------------------------------------------------------
#  Warp meshes — the geometry's adapters to depthai
# ---------------------------------------------------------------------------

def find_warp(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: MountConfig,
    mesh_w: int = WARP_MESH,
    mesh_h: int = WARP_MESH,
) -> list[dai.Point2f]:
    """`warp_mesh_points` as the Warp node's depthai points."""
    return [dai.Point2f(float(x), float(y)) for x, y in warp_mesh_points(
        src_size, out_size, mode_width, mount.fov_h, mount.tilt,
        mount.flip_h, mount.flip_v, mesh_w, mesh_h, mount.lens_fov, mount.lens_center)]


def clip_warp_mesh(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: MountConfig,
) -> tuple[list[dai.Point2f], int, int]:
    """The warp mesh for a raw clip (`clip_warp`): the reprojection and the tilt without the flips and the crop the
    clip already carries; with its mesh size."""
    clip_mount = MountConfig(flip_h=False, flip_v=False,
                            tilt=-mount.tilt if mount.flip_v else mount.tilt,
                            keystone=0.0,           # a recording is already post-keystone
                            fov_h=mount.fov_h,
                            lens_fov=mount.lens_fov,
                            lens_center_x=-mount.lens_center_x if mount.flip_h else mount.lens_center_x,
                            lens_center_y=-mount.lens_center_y if mount.flip_v else mount.lens_center_y)
    mesh: list[dai.Point2f] = find_warp(src_size, out_size, mode_width, clip_mount)
    return mesh, WARP_MESH, WARP_MESH


# ---------------------------------------------------------------------------
#  Keystone (CAMERA.md, the OAK-D)
# ---------------------------------------------------------------------------

KEYSTONE_MESH_W: int = 2
KEYSTONE_MESH_H: int = 64


def find_keystone_warp(width, height, width_offset, flip_h, flip_v, mesh_w, mesh_h) -> list[dai.Point2f]:
    src_points: np.ndarray = np.array([
        [0, 0],             # Top-left
        [width, 0],         # Top-right
        [width, height],    # Bottom-right
        [0, height]         # Bottom-left
    ], dtype=np.float32)

    dst_points: np.ndarray = np.array([
        [width_offset, 0],
        [width - width_offset, 0],
        [width + width_offset, height],
        [-width_offset, height]
    ], dtype=np.float32)

    if flip_h:
        dst_points[:, 0] = width - dst_points[:, 0]
    if flip_v:
        dst_points[:, 1] = height - dst_points[:, 1]

    H: np.ndarray = cv2.getPerspectiveTransform(src_points, dst_points)
    H_inv: np.ndarray = np.linalg.inv(H)

    grid_x: np.ndarray = np.linspace(0, width - 1, mesh_w)
    grid_y: np.ndarray = np.linspace(0, height - 1, mesh_h)

    mesh_points: list[dai.Point2f] = []
    for y in grid_y:
        for x in grid_x:
            p: np.ndarray = np.array([x, y, 1.0])
            src = H_inv @ p
            src /= src[2]  # normalize
            mesh_points.append(dai.Point2f(float(src[0]), float(src[1])))

    return mesh_points


def find_keystone_warp_square(src_width, src_height, square_size, width_offset, flip_h, flip_v, mesh_w, mesh_h) -> list[dai.Point2f]:
    """Create a warp mesh that includes both perspective transformation and cropping to square with optional rotation"""

    square_size = square_size - 1 # -1 to avoid yellow bottom line
    x_offset = (src_width - square_size) / 2
    y_offset = (src_height - square_size) / 2

    # Define the square region we want to extract
    square_corners = np.array([
        [x_offset, y_offset],                    # Top-left
        [x_offset + square_size, y_offset],        # Top-right
        [x_offset + square_size, y_offset + square_size],  # Bottom-right
        [x_offset, y_offset + square_size]         # Bottom-left
    ], dtype=np.float32)

    # These are the destination coordinates (where we map to)
    dst_points = np.array([
        [width_offset, 0],
        [square_size - width_offset, 0],
        [square_size + width_offset, square_size],
        [-width_offset, square_size]
    ], dtype=np.float32)

    if flip_h:
        dst_points[:, 0] = square_size - dst_points[:, 0]
    if flip_v:
        dst_points[:, 1] = square_size - dst_points[:, 1]

    # Calculate transformation matrix
    H = cv2.getPerspectiveTransform(square_corners, dst_points)
    H_inv = np.linalg.inv(H)

    # Generate mesh
    grid_x = np.linspace(0, square_size - 1, mesh_w)
    grid_y = np.linspace(0, square_size - 1, mesh_h)

    mesh_points = []
    for y in grid_y:
        for x in grid_x:
            p = np.array([x, y, 1.0])
            src = H_inv @ p
            src /= src[2]  # normalize
            mesh_points.append(dai.Point2f(float(src[0]), float(src[1])))

    return mesh_points


def _report_lens_reach(src_size: tuple[int, int], mode_width: int, mount: MountConfig) -> None:
    """Log where the lens reaches less than half of `fov` on a side of its center: the edge columns near the
    horizon go black there; over 3 px it warns."""
    if mount.lens_fov <= 0.0:
        return
    focal, cx, _ = source_lens(src_size, mode_width, mount.fov_h, mount.lens_fov, mount.lens_center)
    if focal <= 0.0:
        return
    left: float = math.degrees(cx / focal)
    right: float = math.degrees((src_size[0] - 1 - cx) / focal)
    shortfall: float = mount.fov_h / 2.0 - min(left, right)
    if shortfall <= 0.0:
        return
    pixels: float = math.radians(shortfall) * mode_width / math.radians(mount.fov_h)
    message: str = (f"fov {mount.fov_h:.1f} exceeds the lens reach ({left:.2f} deg left, {right:.2f} deg right "
                    f"of its center): about {pixels:.1f} px black at the {'left' if left < right else 'right'} "
                    f"edge near the horizon")
    (logger.warning if pixels > 3.0 else logger.info)(message)


def build_warp_mesh(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: MountConfig,
) -> tuple[list[dai.Point2f], int, int]:
    """The mesh of the one correction a camera uses, (mesh, mesh_w, mesh_h) for `setWarpMesh`: the keystone
    only for a non-zero `keystone` alone, else the warp; both set warns and takes the tilt."""
    if mount.tilt != 0.0 and mount.keystone != 0.0:
        logger.warning("tilt (%.1f) and keystone (%.2f) are both set; they are exclusive — using tilt",
                       mount.tilt, mount.keystone)
    if mount.tilt != 0.0 or mount.keystone == 0.0:
        _report_lens_reach(src_size, mode_width, mount)
        return find_warp(src_size, out_size, mode_width, mount), WARP_MESH, WARP_MESH

    src_w, src_h = src_size
    out_w, out_h = out_size
    if out_w == src_w:                                          # wide: output is the whole frame
        width_offset: float = src_w * 0.5 * mount.keystone
        mesh = find_keystone_warp(src_w, src_h, width_offset, mount.flip_h, mount.flip_v,
                                  KEYSTONE_MESH_W, KEYSTONE_MESH_H)
    else:                                                       # square: cut from the center
        width_offset = src_h * 0.5 * mount.keystone
        mesh = find_keystone_warp_square(src_w, src_h, out_h, width_offset, mount.flip_h, mount.flip_v,
                                         KEYSTONE_MESH_W, KEYSTONE_MESH_H)
    return mesh, KEYSTONE_MESH_W, KEYSTONE_MESH_H