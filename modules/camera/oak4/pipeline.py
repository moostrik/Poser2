"""The depthai 3 pipeline of one OAK4 camera, built in one pass on the camera's own device.

The sensor at the requested size and type, the correction that turns the sensor's frame into the
contract's, and the IMU. A color frame with neither `tilt` nor `keystone` takes no correction at all:
the Camera node crops it itself and delivers NV12, the lowest latency by far (capture to PC 25 ms
against 34 through `ImageManip` in BGR, over USB), and the host converts and mirrors it. Past that,
two corrections as on the OAK-D: `tilt` (and tilt 0, the cylindrical reprojection that makes a column
one azimuth) goes through the `Warp` node with the mesh from `geometry.warp_mesh_points`, which on
this chip warps grey frames only; `keystone` is a four-point transform in `ImageManip`, in colour or
grey. The detector branch is the model on the delivered frame, the device's object tracker on its detections, and the
tracklets, with their keypoints when the model is a pose model. Nothing here is an XLink stream:
the host reads a node's output through the queue `Camera` creates on it.
"""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import logging
import os

import depthai as dai

from .. import CameraResolution, frame_size, WARP_MESH, warp_mesh_points, keystone_corners, square_crop

from .definitions import SENSOR_SOCKET, frame_type, requested_size, IMU_RATE_HZ, nn_frame_type, \
    POSE_CONFIDENCE_THRESHOLD, POSE_OVERLAP_THRESHOLD, POSE_CROP_PADDING, TRACKER_PERSON_LABEL, TRACKER_TYPE

logger = logging.getLogger(__name__)


@dataclass
class MountConfig:
    """How a camera is mounted and what its lens is, as the correction needs it: the OAK-D's
    `WarpConfig`, under the name of what it holds. `tilt` and `keystone` are exclusive per camera
    (`add_correction`); `fov_h` is the azimuth span of the delivered frame, quoted for the
    un-cropped sensor width; the lens fields are `geometry.source_lens`'s."""
    flip_h: bool
    flip_v: bool
    tilt: float
    keystone: float
    fov_h: float
    lens_fov: float = 0.0
    lens_center_x: float = 0.0
    lens_center_y: float = 0.0

    @property
    def lens_center(self) -> tuple[float, float]:
        return (self.lens_center_x, self.lens_center_y)


@dataclass
class PipelineOutputs:
    """What the host reads and writes: the delivered frames, the sensor's control input, the IMU,
    with a model the tracklets, and with the two-model chain the poses of the crops. `flip_on_host`
    says the camera did not mirror what it delivers, frames and boxes alike, so the host must."""
    video: dai.Node.Output
    control: dai.Node.Input
    imu: dai.Node.Output
    tracklets: dai.Node.Output | None = None
    poses: dai.Node.Output | None = None
    flip_on_host: bool = False


def build_pipeline(
    pipeline: dai.Pipeline,
    fps: float,
    color: bool,
    square: bool,
    resolution: CameraResolution,
    mount: MountConfig,
    frame_height: int = 0,
    pose_archive: Path | None = None,
    detector_archive: Path | None = None,
) -> PipelineOutputs:
    """Build the camera's pipeline into `pipeline`, the one bound to its device.

    `frame_height` is the delivered height the tilt correction produces (`delivered_height`),
    0 for the sensor's rows. The archives are the models' NN archives: neither is frames only,
    `pose_archive` alone the one-model chain, `detector_archive` alone boxes only, and both the
    detector plus the pose model on each person's crop.
    Raises `ValueError` for a colour frame with `tilt` (`pick_correction`).
    """
    src_size: tuple[int, int] = requested_size(color, resolution)
    out_size: tuple[int, int] = frame_size(color, resolution, square, frame_height)
    model: Path | None = pose_archive or detector_archive
    correction: Correction = pick_correction(color, mount.tilt, mount.keystone, mount.flip_v)
    logger.info("OAK4 pipeline: %s, %s, %s%s", 'square' if square else 'wide', 'color' if color else 'grey',
                correction.name.lower(), f', model {model.name}' if model else '')

    camera: dai.node.Camera = pipeline.create(dai.node.Camera).build(SENSOR_SOCKET)
    frames: dai.Node.Output
    if correction == Correction.DIRECT:
        frames = camera.requestOutput(out_size, dai.ImgFrame.Type.NV12, dai.ImgResizeMode.CROP, fps=fps)
    else:
        source: dai.Node.Output = camera.requestOutput(src_size, frame_type(color), fps=fps)
        frames = add_correction(pipeline, source, src_size, out_size, color, mount, correction)
    imu: dai.Node.Output = add_imu(pipeline)
    tracklets: dai.Node.Output | None = None
    poses: dai.Node.Output | None = None
    if pose_archive and detector_archive:
        tracklets, poses = add_pose_chain(pipeline, frames, fps, detector_archive, pose_archive)
    elif model:
        tracklets = add_one_model_chain(pipeline, frames, model)
    return PipelineOutputs(video=frames, control=camera.inputControl, imu=imu, tracklets=tracklets, poses=poses,
                           flip_on_host=correction == Correction.DIRECT and mount.flip_h)


class Correction(IntEnum):
    WARP = 0            # the mesh: the tilt, or at tilt 0 the cylindrical reprojection
    FOUR_POINT = 1      # the keystone; at keystone 0 the plain crop and flips
    DIRECT = 2          # none: the Camera node's own crop, NV12, mirrored on the host


def pick_correction(color: bool, tilt: float, keystone: float, flip_v: bool) -> Correction:
    """The one correction a camera uses, picked as `oak.pipeline.build_warp_mesh` picks it where the
    chip allows: both set takes `tilt`; both zero is the reprojection in grey. The chip warps grey
    frames only, so a colour frame with both zero needs no reprojection at all and goes `DIRECT`,
    unless `flip_v` asks for a flip the host does not do, which stays on the four-point transform.
    Raises `ValueError` for a colour frame with `tilt`, which the chip cannot correct: the camera
    then does not open, rather than deliver a frame the tracker would read wrong."""
    if tilt != 0.0 and keystone != 0.0:
        logger.warning("tilt (%.1f) and keystone (%.2f) are both set; they are exclusive, using tilt", tilt, keystone)
    if tilt != 0.0:
        if color:
            raise ValueError('the OAK4 warps grey frames only: tilt needs color off, or keystone instead')
        return Correction.WARP
    if keystone == 0.0 and not color:
        return Correction.WARP
    if keystone == 0.0 and not flip_v:
        return Correction.DIRECT
    return Correction.FOUR_POINT


def add_correction(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int],
                   out_size: tuple[int, int], color: bool, mount: MountConfig, correction: Correction) -> dai.Node.Output:
    """The correction `pick_correction` named, on the chip."""
    if correction == Correction.WARP:
        return add_warp(pipeline, source, src_size, out_size, mount)
    if mount.keystone == 0.0:
        logger.info("OAK4 colour frame: cropped and flipped, not reprojected (the chip warps grey frames only)")
    return add_keystone(pipeline, source, src_size, out_size, color, mount)


def add_warp(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int],
             out_size: tuple[int, int], mount: MountConfig) -> dai.Node.Output:
    """The warp from the sensor's frame to the delivered one, the mesh from the shared geometry."""
    mesh: list[tuple[float, float]] = [(float(x), float(y)) for x, y in warp_mesh_points(
        src_size, out_size, src_size[0], mount.fov_h, mount.tilt, mount.flip_h, mount.flip_v,
        WARP_MESH, WARP_MESH, mount.lens_fov, mount.lens_center)]
    warp: dai.node.Warp = pipeline.create(dai.node.Warp)
    warp.setMaxOutputFrameSize(out_size[0] * out_size[1])
    warp.setOutputSize(*out_size)
    warp.setWarpMesh(mesh, WARP_MESH, WARP_MESH)
    source.link(warp.inputImage)
    return warp.out


def add_keystone(pipeline: dai.Pipeline, source: dai.Node.Output, src_size: tuple[int, int],
                 out_size: tuple[int, int], color: bool, mount: MountConfig) -> dai.Node.Output:
    """The keystone as a four-point transform: the frame's corners pinned to the points the
    OAK-D's homography pins (`geometry.keystone_corners`), the square cut from the centre first."""
    manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
    initial: dai.ImageManipConfig = manip.initialConfig
    if out_size[0] == src_size[0]:                          # wide: the whole frame
        src, dst = keystone_corners(src_size[0], src_size[1], mount.keystone, mount.flip_h, mount.flip_v)
    else:                                                   # square: cut from the centre
        size: int = out_size[1]
        initial.addCrop(*square_crop(src_size, size))
        src, dst = keystone_corners(size, size, mount.keystone, mount.flip_h, mount.flip_v, offset_base=src_size[1])
    initial.addTransformFourPoints([dai.Point2f(x, y) for x, y in src], [dai.Point2f(x, y) for x, y in dst], False)
    initial.setOutputSize(out_size[0], out_size[1], dai.ImageManipConfig.ResizeMode.STRETCH)
    initial.setFrameType(frame_type(color))
    manip.setMaxOutputFrameSize(out_size[0] * out_size[1] * (3 if color else 1))
    source.link(manip.inputImage)
    return manip.out


def add_one_model_chain(pipeline: dai.Pipeline, frames: dai.Node.Output, archive_path: Path) -> dai.Node.Output:
    """The model on the delivered frame, the device's object tracker on its detections, and the
    tracklets out. The archive's config tells the network how to decode them, so no decode of ours:
    a pose model's boxes bring their 17 keypoints in each tracklet's source detection, a person
    detector's only the box."""
    network = add_detector(pipeline, frames, dai.NNArchive(archive_path))
    return add_tracker(pipeline, network.passthrough, network.out)


def add_detector(pipeline: dai.Pipeline, frames: dai.Node.Output, archive: dai.NNArchive) -> dai.node.DetectionNetwork:
    """The delivered frame stretched into the model and the network on it: the first stage of every
    chain. STRETCHED, never cropped or letterboxed, as the OAK-D's detector takes it: nothing is
    lost at the edges and the normalised boxes map 1:1 onto the delivered frame; an aspect mismatch
    costs detection quality, never geometry."""
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
    """The two-model chain, Luxonis's way (depthai-nodes): the detector on the stretched frame, its
    detections filtered to the people, one letterboxed crop per person cut out of the delivered
    frame on the device (`FrameCropper`), the top-down pose model on each crop with its parser
    (`ParsingNeuralNetwork`), and the crops' keypoints gathered per frame beside the detections
    they came from (`GatherData`). The tracker runs on the same people. Returns the tracklets and
    the gathered poses; `Camera` joins the two by frame.

    The parser's own score threshold is turned off: it would drop a keypoint below it from the
    message and leave the rest unnumbered, while the contract carries all 17 with their scores.
    """
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
