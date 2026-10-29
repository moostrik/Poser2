"""The depthai pipeline of one OAK camera, built in one pass.

A source (the sensor, or an `XLinkIn` the simulator feeds with clips), the warp that turns the
sensor's frame into the contract's (`geometry.warp_mesh_points`, or the keystone of the older
installations), the video output, and with `yolo` the detector branch: a manip that sizes the
frame for the blob, the YOLO network, the device's object tracker and the tracklet output. A
device also gets the IMU. Each part is one function; the simulation differs in its source, and in
whether the warp is applied (`warp_clips`: a raw clip goes through it, a recorded one is already
warped). The warp mesh builders at the end are the geometry's adapters to depthai.
"""

from dataclasses import dataclass
from pathlib import Path
import logging
import math

import cv2
import depthai as dai
import numpy as np

from .. import CameraResolution, mode_size, frame_size, WARP_MESH, warp_mesh_points, source_lens

from .definitions import (
    YOLOV8_WIDE_6S, YOLOV8_WIDE_7S, YOLOV8_SQUARE_6S, YOLOV8_SQUARE_7S,
    YOLO_CONFIDENCE_THRESHOLD, YOLO_OVERLAP_THRESHOLD, detector_input_size,
    TRACKER_PERSON_LABEL, TRACKER_TYPE, mono_mode, color_mode, IMU_RATE_HZ,
)

logger = logging.getLogger(__name__)


# The XLink streams between the device and the host, named once for both ends.
VIDEO_STREAM: str = 'video'             # the delivered frames
TRACKLETS_STREAM: str = 'tracklets'     # the object tracker's output
IMU_STREAM: str = 'imu'                 # the accelerometer, on a device
CONTROL_STREAM: str = 'control'         # the sensor controls, on a device
CLIPS_STREAM: str = 'clips'             # the simulator's clip frames, in


@dataclass
class WarpConfig:
    """How a camera is mounted and what its lens is, as the warp needs it.

    `tilt` (degrees, positive = aimed up) re-aims the camera so a column reads as one azimuth,
    at the cost of the frame edges. `keystone` (fraction of the frame) squares up a person seen
    from a tilted camera while keeping the whole frame. They are exclusive per camera — see
    `build_warp_mesh`. `fov_h` is the azimuth span of the delivered frame, quoted for the
    un-cropped sensor width. `lens_fov`, `lens_centre_x`, `lens_centre_y` are the lens itself
    (`geometry.source_lens`); at their defaults the lens is taken to be `fov_h`.
    """
    flip_h: bool
    flip_v: bool
    tilt: float
    keystone: float
    fov_h: float
    lens_fov: float = 0.0
    lens_centre_x: float = 0.0
    lens_centre_y: float = 0.0

    @property
    def lens_centre(self) -> tuple[float, float]:
        return (self.lens_centre_x, self.lens_centre_y)


def model_path(models: str, square: bool, clips: bool) -> Path:
    """The detector blob for this frame shape. The simulator's blob has one shave more: with no
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
    mount: WarpConfig,
    frame_height: int = 0,
    clips: bool = False,
    warp_clips: bool = False,
) -> None:
    """Build the camera's pipeline into `pipeline`.

    `clips` takes the frames from the host (the simulator) instead of the sensor, at the delivered
    size, or at the sensor's size with `warp_clips`, when they go through the warp like the
    sensor's frames. `frame_height` is the delivered height the warp produces (`delivered_height`),
    0 for the sensor mode's rows.
    """
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
    """The sensor at its mode, with its control stream; returns its frame output.

    Colour comes off the `preview` output, planar BGR at the mode size; mono off the left camera's
    `out`, the one sensor a mono installation uses."""
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
    """The simulator's frames, in. The ceiling takes a clip at the sensor's size or the delivered one."""
    channels: int = 3 if color else 1
    clips: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    clips.setStreamName(CLIPS_STREAM)
    clips.setMaxDataSize(max(src_size[0] * src_size[1], out_size[0] * out_size[1]) * channels)
    return clips.out


def add_warp(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int], out_size: tuple[int, int],
             color: bool, mount: WarpConfig, clips: bool) -> dai.Node.Output:
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
    """The detector branch: the frame sized for the blob, YOLO, the device's object tracker on
    its detections, and the tracklet output.

    The manip stretches a wide frame into the blob (`detector_input_size` explains why nothing is
    cropped) and keeps the aspect of a square one, which is the blob's. The tracker takes the
    network's passthrough as its frame, so it only ever sees frames that were inferred.
    """
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
    network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
    network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
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
    """A slow accelerometer feed, so the camera can report how it is actually mounted.

    `ACCELEROMETER_RAW` rather than `GRAVITY`: gravity is a fused output that needs the IMU's own
    DSP, which the BNO086 boards have and the raw BMI270 ones do not. A camera on a tripod is
    static, so the raw accelerometer *is* the gravity vector once it is averaged, and this works
    on every board.

    A few hertz is plenty — nothing in the show reads this, it exists for whoever is aiming the
    cameras. The node is created unconditionally; a board without an IMU simply never sends, and
    `Camera._setup_imu_queue` only subscribes when `getConnectedIMU` names a sensor.
    """
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
    mount: WarpConfig,
    mesh_w: int = WARP_MESH,
    mesh_h: int = WARP_MESH,
) -> list[dai.Point2f]:
    """The Warp node's mesh for the frame contract — column = azimuth, row = tangent of
    elevation, tilt undone — as depthai points.

    All of the geometry lives in `modules.camera.geometry` (`warp_mesh_points`), which has no depthai or
    OpenCV dependency and carries the explanation of the lens model, the window and the mesh
    density. This is only the adapter. For the *other* correction, `keystone`, see
    `find_keystone_warp`.
    """
    return [dai.Point2f(float(x), float(y)) for x, y in warp_mesh_points(
        src_size, out_size, mode_width, mount.fov_h, mount.tilt,
        mount.flip_h, mount.flip_v, mesh_w, mesh_h, mount.lens_fov, mount.lens_centre)]


def clip_warp_mesh(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: WarpConfig,
) -> tuple[list[dai.Point2f], int, int]:
    """The warp mesh for a RECORDING rather than a sensor, ready to splat into `setWarpMesh`.

    Recordings come off `output_video`, which is **after** the warp, so a clip already carries
    the flips — and the square crop — that were applied when it was shot. Re-applying the live
    mesh would do all of that a second time, un-mirroring the image and cropping an already
    cropped frame. What is wanted is only the *difference*, which is the reprojection and the
    tilt on their own:

    - the flips come off, because they are already in the pixels;
    - the clip is the source at its own (sensor) size, because the crop is already in the
      pixels; the output is the delivered size, as tall as `delivered_height` makes it;
    - the tilt's sign flips under `flip_v`, because a vertical mirror reverses a rotation about
      the horizontal axis, while a horizontal mirror commutes with it and needs no correction;
    - the lens centre offset mirrors with the pixels: `x` under `flip_h`, `y` under `flip_v`.

    ASSUMES THE CLIP IS RAW: equidistant, shot at `tilt = 0`, before the warp reprojected its
    output. That holds for every recording made before `tilt` existed, and those were shot on
    the same units the shared lens describes. A clip recorded since is already reprojected and
    would be reprojected twice; capture-time geometry is still not stored beside clips — see
    the plan's "save the preset alongside each recording".
    """
    clip_mount = WarpConfig(flip_h=False, flip_v=False,
                            tilt=-mount.tilt if mount.flip_v else mount.tilt,
                            keystone=0.0,           # a recording is already post-keystone
                            fov_h=mount.fov_h,
                            lens_fov=mount.lens_fov,
                            lens_centre_x=-mount.lens_centre_x if mount.flip_h else mount.lens_centre_x,
                            lens_centre_y=-mount.lens_centre_y if mount.flip_v else mount.lens_centre_y)
    mesh: list[dai.Point2f] = find_warp(src_size, out_size, mode_width, clip_mount)
    return mesh, WARP_MESH, WARP_MESH


# ---------------------------------------------------------------------------
#  Keystone — the pre-existing correction, kept verbatim
# ---------------------------------------------------------------------------
# These two builders are the original `find_perspective_warp` / `_square`, unchanged apart
# from the name, at their original 2 x 64 mesh. They build a homography that pins all four
# corners of the output to the frame corners: a rotation composed with a stretch-to-fit. That
# is the wrong model for re-aiming a wide lens (see `geometry.warp_mesh_points`), but it is
# exactly what hd_trio and deep_flow rely on — a person seen from a down-tilted camera gets
# normal proportions while the whole frame is kept. Two columns suffice because the map is
# affine in x per row; the vertical curvature is what the 64 rows are for. Nothing here may
# change without those installations changing with it.

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


def _report_lens_reach(src_size: tuple[int, int], mode_width: int, mount: WarpConfig) -> None:
    """`fov` is the frame's span, and the lens has to reach half of it on EACH side of its own
    centre — which is not the frame centre. Where it falls short, the edge columns near the
    horizon read past the sensor and go black. A shortfall of a pixel or two is the price of a
    shared lens centre and is logged as information; more is a preset to fix, and warns."""
    if mount.lens_fov <= 0.0:
        return
    focal, cx, _ = source_lens(src_size, mode_width, mount.fov_h, mount.lens_fov, mount.lens_centre)
    if focal <= 0.0:
        return
    left: float = math.degrees(cx / focal)
    right: float = math.degrees((src_size[0] - 1 - cx) / focal)
    shortfall: float = mount.fov_h / 2.0 - min(left, right)
    if shortfall <= 0.0:
        return
    pixels: float = math.radians(shortfall) * mode_width / math.radians(mount.fov_h)
    message: str = (f"fov {mount.fov_h:.1f} exceeds the lens reach ({left:.2f} deg left, {right:.2f} deg right "
                    f"of its centre): about {pixels:.1f} px black at the {'left' if left < right else 'right'} "
                    f"edge near the horizon")
    (logger.warning if pixels > 3.0 else logger.info)(message)


def build_warp_mesh(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: WarpConfig,
) -> tuple[list[dai.Point2f], int, int]:
    """Pick the one correction a camera uses and return (mesh, mesh_w, mesh_h) for `setWarpMesh`.

    `tilt` and `keystone` are exclusive: a keystone spreads columns by height, which is exactly
    what tilt removes so a column can mean one azimuth, and tilt shifts the frame, which is what
    keystone exists to avoid. Both non-zero is not a valid state for any camera, so it warns and
    takes `tilt`. Both zero goes through the tilt path, which is NOT the identity: it is the
    equidistant -> cylindrical reprojection that makes a column one azimuth (see
    `geometry.warp_mesh_points`). Only a non-zero `keystone` reaches the old builder.
    """
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
    else:                                                       # square: cut from the centre
        width_offset = src_h * 0.5 * mount.keystone
        mesh = find_keystone_warp_square(src_w, src_h, out_h, width_offset, mount.flip_h, mount.flip_v,
                                         KEYSTONE_MESH_W, KEYSTONE_MESH_H)
    return mesh, KEYSTONE_MESH_W, KEYSTONE_MESH_H