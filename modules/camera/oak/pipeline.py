"""The depthai pipeline of one OAK-D camera: its source (the sensor or clips), the warp, the video output, the
detector and tracker, and the IMU."""

from pathlib import Path
import logging

import depthai as dai

from ..geometry import CameraResolution, IMU_RATE_HZ, MountConfig, mode_size, frame_size

from .models import YOLOV8_WIDE_6S, YOLOV8_WIDE_7S, YOLOV8_SQUARE_6S, YOLOV8_SQUARE_7S, \
    DETECTOR_CONFIDENCE_THRESHOLD, DETECTOR_OVERLAP_THRESHOLD, detector_input_size, TRACKER_PERSON_LABEL, TRACKER_TYPE
from .sensor import mono_mode, color_mode
from .warp_mesh import build_warp_mesh, clip_warp_mesh

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
