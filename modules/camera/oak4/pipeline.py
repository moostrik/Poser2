"""The depthai 3 pipeline of one OAK4 camera on its own device: the color frame (cropped, undistorted, NV12) or the
grey one through the warp, the model chain and its tracker, Neural Depth aligned to the frame, and the IMU
(OAK4.md, the frame)."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import logging
import os

import depthai as dai

from .. import CAPTURE_TOLERANCE
from ..geometry import CameraResolution, frame_size, WARP_MESH, warp_mesh_points

from .device import SYSTEM_RATE_HZ
from .imu import IMU_RATE_HZ
from .models import nn_frame_type, POSE_CONFIDENCE_THRESHOLD, POSE_OVERLAP_THRESHOLD, POSE_CROP_PADDING, TRACKER_PERSON_LABEL, \
    TRACKER_TYPE
from .sensor import SENSOR_SOCKET, STEREO_SOCKETS, STEREO_SIZE, DEPTH_MODELS, DepthModel, requested_size, direct_size

logger = logging.getLogger(__name__)

DEPTH_STREAM: str = 'depth'             # the depth's name in the depth's message group
CONFIDENCE_STREAM: str = 'confidence'   # its confidence's


@dataclass
class MountConfig:
    """How a camera is mounted and what its lens is, as the correction needs it."""
    flip_h: bool
    flip_v: bool
    fov_h: float                # ° the delivered frame spans, quoted for the un-cropped sensor width; the lens `source_lens`'s
    lens_fov: float = 0.0
    lens_center_x: float = 0.0
    lens_center_y: float = 0.0

    @property
    def lens_center(self) -> tuple[float, float]:
        return (self.lens_center_x, self.lens_center_y)


@dataclass
class PipelineOutputs:
    """The pipeline's outputs and inputs the host reads and writes; `flip_h_on_host` and `flip_v_on_host`: the flips
    the camera leaves to the frame's flags, for its frames and boxes alike (a color frame's; the grey warp flips)."""
    video: dai.Node.Output
    control: dai.Node.Input
    imu: dai.Node.Output
    system: dai.Node.Output
    tracklets: dai.Node.Output | None = None
    poses: dai.Node.Output | None = None
    depth: dai.Node.Output | None = None
    depth_config: dai.Node.Input | None = None
    flip_h_on_host: bool = False
    flip_v_on_host: bool = False


def delivered_size(color: bool, resolution: CameraResolution, frame_height: int = 0) -> tuple[int, int]:
    """The delivered frame: color 5:4, as the Camera node crops it, grey the sensor mode's full width."""
    return direct_size(resolution) if color else frame_size(color, resolution, False, frame_height)


def build_pipeline(
    pipeline: dai.Pipeline,
    fps: float,
    color: bool,
    resolution: CameraResolution,
    mount: MountConfig,
    frame_height: int = 0,
    pose_archive: Path | None = None,
    detector_archive: Path | None = None,
    depth_model: DepthModel = DepthModel.OFF,
) -> PipelineOutputs:
    """Build the camera's pipeline into `pipeline`, the one bound to its device; the archives set the model chain
    (`model_archives`), `frame_height` is the delivered height, 0 the sensor's rows."""
    src_size: tuple[int, int] = requested_size(color, resolution)
    out_size: tuple[int, int] = delivered_size(color, resolution, frame_height)
    model: Path | None = pose_archive or detector_archive
    logger.info("OAK4 pipeline: %dx%d, %s%s%s", out_size[0], out_size[1], 'color' if color else 'grey',
                f', model {model.name}' if model else '', f', depth {depth_model.name}' if depth_model != DepthModel.OFF else '')

    camera: dai.node.Camera = pipeline.create(dai.node.Camera).build(SENSOR_SOCKET)
    frames: dai.Node.Output
    if color:
        frames = camera.requestOutput(out_size, dai.ImgFrame.Type.NV12, dai.ImgResizeMode.CROP, fps=fps, enableUndistortion=True)
    else:
        frames = add_warp(pipeline, camera.requestOutput(src_size, dai.ImgFrame.Type.GRAY8, fps=fps), src_size, out_size, mount)
    imu: dai.Node.Output = add_imu(pipeline)
    tracklets: dai.Node.Output | None = None
    poses: dai.Node.Output | None = None
    if pose_archive and detector_archive:
        tracklets, poses = add_pose_chain(pipeline, frames, fps, detector_archive, pose_archive)
    elif model:
        tracklets = add_one_model_chain(pipeline, frames, model)
    depth: dai.Node.Output | None = None
    depth_config: dai.Node.Input | None = None
    if depth_model != DepthModel.OFF:
        depth, depth_config = add_depth(pipeline, fps, depth_model)
    return PipelineOutputs(video=frames, control=camera.inputControl, imu=imu, system=add_system_logger(pipeline),
                           tracklets=tracklets, poses=poses, depth=depth, depth_config=depth_config,
                           flip_h_on_host=color and mount.flip_h, flip_v_on_host=color and mount.flip_v)


def add_warp(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int],
             out_size: tuple[int, int], mount: MountConfig) -> dai.Node.Output:
    """The warp from the sensor's frame to the delivered one, the mesh from the shared geometry."""
    mesh: list[tuple[float, float]] = [(float(x), float(y)) for x, y in warp_mesh_points(
        src_size, out_size, src_size[0], mount.fov_h, 0.0, mount.flip_h, mount.flip_v,
        WARP_MESH, WARP_MESH, mount.lens_fov, mount.lens_center)]
    warp: dai.node.Warp = pipeline.create(dai.node.Warp)
    warp.setMaxOutputFrameSize(out_size[0] * out_size[1])
    warp.setOutputSize(*out_size)
    warp.setWarpMesh(mesh, WARP_MESH, WARP_MESH)
    source.link(warp.inputImage)
    return warp.out


def add_one_model_chain(pipeline: dai.Pipeline, frames: dai.Node.Output, archive_path: Path) -> dai.Node.Output:
    """The model on the delivered frame, decoded by its archive's config, and the device's tracker on its detections;
    returns the tracklets, with a pose model's keypoints in their source detections."""
    network = add_detector(pipeline, frames, dai.NNArchive(archive_path))
    return add_tracker(pipeline, network.passthrough, network.out)


def add_detector(pipeline: dai.Pipeline, frames: dai.Node.Output, archive: dai.NNArchive) -> dai.node.DetectionNetwork:
    """The delivered frame stretched into the model and the network on it: the first stage of every
    chain. STRETCHED, never cropped or letterboxed: nothing is lost at the edges and the normalised
    boxes map 1:1 onto the delivered frame; an aspect mismatch costs detection quality, never geometry."""
    size: tuple[int, int] | None = archive.getInputSize()
    if size is None:
        raise ValueError('the model archive names no input size')
    manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(size[0], size[1], dai.ImageManipConfig.ResizeMode.STRETCH)
    manip.initialConfig.setFrameType(nn_frame_type(archive))
    manip.setMaxOutputFrameSize(size[0] * size[1] * 3)
    frames.link(manip.inputImage)

    network: dai.node.DetectionNetwork = pipeline.create(dai.node.DetectionNetwork)
    network.setNNArchive(archive)
    network.setConfidenceThreshold(POSE_CONFIDENCE_THRESHOLD)
    network.detectionParser.setIouThreshold(POSE_OVERLAP_THRESHOLD)
    network.input.setBlocking(False)
    manip.out.link(network.input)
    return network


def add_tracker(pipeline: dai.Pipeline, frames: dai.Node.Output, detections: dai.Node.Output) -> dai.Node.Output:
    """The device's object tracker on the people; `frames` are the frames the detections were made
    on (the network's passthrough), so it only ever sees frames that were inferred."""
    tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
    tracker.setTrackerType(TRACKER_TYPE)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
    frames.link(tracker.inputTrackerFrame)
    frames.link(tracker.inputDetectionFrame)
    detections.link(tracker.inputDetections)
    return tracker.out


def add_pose_chain(pipeline: dai.Pipeline, frames: dai.Node.Output, fps: float, detector_archive: Path,
                   pose_archive: Path) -> tuple[dai.Node.Output, dai.Node.Output]:
    """The two-model chain (depthai-nodes): the detector, a crop per player cut on the device, the top-down pose
    model on each, and the crops' keypoints gathered per frame beside their detections; returns the tracklets and
    the gathered keypoints, which `KeypointJoin` puts together. The parser's score threshold is off, as it would
    drop keypoints and leave the rest unnumbered."""
    level: str | None = os.environ.pop('DEPTHAI_LEVEL', None)   # depthai_nodes rejects depthai's `error` at import
    try:
        from depthai_nodes.node import ImgDetectionsFilter, FrameCropper, ParsingNeuralNetwork, GatherData
    finally:
        if level is not None:
            os.environ['DEPTHAI_LEVEL'] = level

    detector = add_detector(pipeline, frames, dai.NNArchive(detector_archive))
    persons: ImgDetectionsFilter = pipeline.create(ImgDetectionsFilter).build(detector.out)
    persons.keepLabels([TRACKER_PERSON_LABEL])

    pose = dai.NNArchive(pose_archive)
    crop_size: tuple[int, int] | None = pose.getInputSize()
    if crop_size is None:
        raise ValueError(f'{pose_archive.name} names no input size')
    cropper: FrameCropper = pipeline.create(FrameCropper).fromImgDetections(
        persons.out, outputSize=crop_size, resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        padding=POSE_CROP_PADDING).build(frames)
    network: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(cropper.out, pose)
    network.getParser(0).setScoreThreshold(0.0)  # pyright: ignore[reportAttributeAccessIssue]
    gather: GatherData = pipeline.create(GatherData).build(int(round(fps)), network.out, persons.out)

    tracklets: dai.Node.Output = add_tracker(pipeline, detector.passthrough, persons.out)
    return tracklets, gather.out


def add_depth(pipeline: dai.Pipeline, fps: float, model: DepthModel) -> tuple[dai.Node.Output, dai.Node.Input]:
    """Neural Depth off the stereo pair, on the left camera's view, aligned on the GPU and not by `ImageAlign` (OAK4.md,
    depth); returns one message group a frame, its depth (`DEPTH_STREAM`) with its confidence (`CONFIDENCE_STREAM`),
    and the config input."""
    left, right = [pipeline.create(dai.node.Camera).build(socket, sensorFps=fps).requestOutput(STEREO_SIZE, fps=fps)
                   for socket in STEREO_SOCKETS]
    neural: dai.node.NeuralDepth = pipeline.create(dai.node.NeuralDepth).build(left, right, DEPTH_MODELS[model])
    sync: dai.node.Sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=CAPTURE_TOLERANCE))
    neural.depth.link(sync.inputs[DEPTH_STREAM])
    neural.confidence.link(sync.inputs[CONFIDENCE_STREAM])
    return sync.out, neural.inputConfig


def add_imu(pipeline: dai.Pipeline) -> dai.Node.Output:
    """A slow accelerometer feed, so the camera can report how it is mounted. Raw rather than the
    fused gravity, averaged on the host: a camera on a tripod is static, so the average is the
    gravity vector, and this needs nothing of the IMU's own fusion."""
    imu: dai.node.IMU = pipeline.create(dai.node.IMU)
    accelerometer = dai.IMUSensorConfig()
    accelerometer.sensorId = dai.IMUSensor.ACCELEROMETER_RAW
    accelerometer.reportRate = IMU_RATE_HZ
    imu.enableIMUSensor(accelerometer)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    return imu.out


def add_system_logger(pipeline: dai.Pipeline) -> dai.Node.Output:
    """The camera's own report of its CPU, memory and temperature. It has no power and no NPU load."""
    system: dai.node.SystemLogger = pipeline.create(dai.node.SystemLogger)
    system.setRate(SYSTEM_RATE_HZ)
    return system.out
