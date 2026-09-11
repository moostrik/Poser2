from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import logging

import cv2
import depthai as dai
import numpy as np

from .definitions import (
    FrameType,
    YOLOV8_WIDE_5S, YOLOV8_WIDE_6S, YOLOV8_WIDE_7S,
    YOLOV8_SQUARE_5S, YOLOV8_SQUARE_6S, YOLOV8_SQUARE_7S,
    YOLO_CONFIDENCE_THRESHOLD, YOLO_OVERLAP_THRESHOLD,
    DETECTOR_INPUT_WIDE, DETECTOR_INPUT_SQUARE,
    TRACKER_PERSON_LABEL, TRACKER_TYPE,
    DEPTH_TRACKER_BOX_SCALE, DEPTH_TRACKER_LOCATION,
    DEPTH_TRACKER_MIN_DEPTH, DEPTH_TRACKER_MAX_DEPTH,
    CameraResolution, mono_mode, color_mode, mono_frame_size, color_frame_size,
    WARP_MESH, equirect_mesh_points, IMU_RATE_HZ,
)

logger = logging.getLogger(__name__)


@dataclass
class WarpConfig:
    """How a camera is mounted, as the warp needs it.

    `tilt` (degrees, positive = aimed up) re-aims the camera so a column reads as one azimuth,
    at the cost of the frame edges. `keystone` (fraction of the frame) squares up a person seen
    from a tilted camera while keeping the whole frame. They are exclusive per camera — see
    `build_warp_mesh`. `fov_h` is the horizontal field of the un-cropped sensor mode, which is
    what turns the tilt angle into pixels.
    """
    flip_h: bool
    flip_v: bool
    tilt: float
    keystone: float
    fov_h: float

def get_frame_types(do_color: bool, do_stereo: bool, show_stereo: bool, simulate: bool) -> list[FrameType]:
    frame_types: list[FrameType] = [FrameType.NONE_]
    frame_types.append(FrameType.VIDEO)
    if do_stereo:
        if not simulate:
            frame_types.append(FrameType.LEFT_)
            frame_types.append(FrameType.RIGHT)
        if show_stereo:
            frame_types.append(FrameType.DEPTH)
    return frame_types

def get_stereo_config(do_color: bool) -> dai.RawStereoDepthConfig:
    stereoConfig: dai.RawStereoDepthConfig = dai.RawStereoDepthConfig()
    if do_color:
        stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
    else:
        stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
    return stereoConfig

def add_imu_node(pipeline: dai.Pipeline) -> dai.node.IMU:
    """A slow accelerometer feed, so the camera can report how it is actually mounted.

    `ACCELEROMETER_RAW` rather than `GRAVITY`: gravity is a fused output that needs the IMU's own
    DSP, which the BNO086 boards have and the raw BMI270 ones do not. A camera on a tripod is
    static, so the raw accelerometer *is* the gravity vector once it is averaged, and this works
    on every board.

    A few hertz is plenty — nothing in the show reads this, it exists for whoever is aiming the
    cameras. The node is created unconditionally; a board without an IMU simply never sends, and
    `Camera._setup_queues` only subscribes when `getConnectedIMU` names a sensor.
    """
    imu: dai.node.IMU = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, IMU_RATE_HZ)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    imu_out: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
    imu_out.setStreamName('imu')
    imu.out.link(imu_out.input)
    return imu


def get_model_path(model_path: str, square: bool, stereo: bool, simulate: bool) -> Path:
    if square:
        if stereo:
            return (Path(model_path) / YOLOV8_SQUARE_5S).resolve().absolute()
        elif simulate:
            return (Path(model_path) / YOLOV8_SQUARE_7S).resolve().absolute()
        return (Path(model_path) / YOLOV8_SQUARE_6S).resolve().absolute()
    if stereo:
        return (Path(model_path) / YOLOV8_WIDE_5S).resolve().absolute()
    elif simulate:
        return (Path(model_path) / YOLOV8_WIDE_7S).resolve().absolute()
    return (Path(model_path) / YOLOV8_WIDE_6S).resolve().absolute()


def setup_pipeline(
    pipeline : dai.Pipeline,
    model_path:str,
    fps: float = 30.0,
    square: bool = True,
    do_color: bool = True,
    do_stereo: bool = True,
    do_yolo: bool = True,
    resolution: CameraResolution = CameraResolution.P800,
    show_stereo: bool = False,
    mount: WarpConfig = WarpConfig(False, False, 0.0, 0.0, 127.0),
    simulate: bool = False,
    warp_clips: bool = False
    ) -> None:

    if square and do_stereo:
        logger.info("Square mode is not compatible with stereo depth. Setting to Wide mode.")
        square = False

    options: list[str] = [
        'Square,' if square else 'Wide,',
        'Color,' if do_color else 'Mono,',
        'Stereo (Show),' if do_stereo and show_stereo else 'Stereo (Hidden),' if do_stereo else '',
        'Yolo,' if do_yolo else '',
        'Simulate' if simulate else ''
    ]

    pipeline_description = "Depth Pipeline: " + " ".join(filter(None, options))
    logger.info(pipeline_description)

    nn_path: Path = get_model_path(model_path, square, do_stereo, simulate)
    if not simulate:
        if do_color:
            if do_stereo:
                if do_yolo:
                    SetupColorStereoYolo(pipeline, fps, resolution, show_stereo, nn_path)
                else:
                    SetupColorStereo(pipeline, fps, resolution, show_stereo = True)
            else:
                if do_yolo:
                    SetupColorYolo(pipeline, fps, resolution, square, mount, nn_path)
                else:
                    SetupColor(pipeline, fps, resolution, square, mount)
        else:
            if do_stereo:
                if do_yolo:
                    SetupMonoStereoYolo(pipeline, fps, resolution, show_stereo, nn_path)
                else:
                    SetupMonoStereo(pipeline, fps, resolution, show_stereo = True)
            else:
                if do_yolo:
                    SetupMonoYolo(pipeline, fps, resolution, square, mount, nn_path)
                else:
                    SetupMono(pipeline, fps, resolution, square, mount)
    else:
        if do_color:
            if do_stereo:
                if do_yolo:
                    SimulationColorStereoYolo(pipeline, fps, resolution, show_stereo, nn_path)
                else:
                    SimulationColorStereo(pipeline, fps, resolution, show_stereo)
            else:
                if do_yolo:
                    SimulationColorYolo(pipeline, fps, resolution, square, mount, nn_path, warp_clips)
                else:
                    SimulationColor(pipeline, fps, resolution, square, mount, warp_clips)
        else:
            if do_stereo:
                if do_yolo:
                    SimulationMonoStereoYolo(pipeline, fps, resolution, show_stereo, nn_path)
                else:
                    SimulationMonoStereo(pipeline, fps, resolution, show_stereo)
            else:
                if do_yolo:
                    SimulationMonoYolo(pipeline, fps, resolution, square, mount, nn_path, warp_clips)
                else:
                    SimulationMono(pipeline, fps, resolution, square, mount, warp_clips)


class Setup():
    def __init__(self, pipeline : dai.Pipeline, fps: float) -> None:
        self.pipeline: dai.Pipeline = pipeline
        self.fps: float = fps

class SetupColor(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution, square: bool, mount: WarpConfig) -> None:
        super().__init__(pipeline, fps)

        self.resolution: CameraResolution = resolution
        self.sensor_mode: dai.ColorCameraProperties.SensorResolution = color_mode(resolution)
        self.width, self.height = color_frame_size(resolution)

        self.data_size: int = self.width * self.height * 3


        self.color: dai.node.ColorCamera = pipeline.create(dai.node.ColorCamera)
        self.color.setResolution(self.sensor_mode)
        self.color.setFps(self.fps)
        self.color.setInterleaved(False)
        self.color.setPreviewSize(self.width, self.height)

        self.color_warp: dai.node.Warp = pipeline.create(dai.node.Warp)
        self.mode_width: int = self.width   # `fov_h` is quoted for the un-cropped width
        self.mount: WarpConfig = mount

        if square:
            self.width = self.height        # make square
            self.data_size = self.width * self.height * 3

        warp_mesh, mesh_w, mesh_h = build_warp_mesh(
            (self.mode_width, self.height), (self.width, self.height), self.mode_width, mount)

        self.color_warp.setMaxOutputFrameSize(self.data_size)
        self.color_warp.setOutputSize(self.width, self.height)

        self.color_warp.setWarpMesh(warp_mesh, mesh_w, mesh_h)
        self.color.preview.link(self.color_warp.inputImage)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.color_warp.out.link(self.output_video.input)

        self.color_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.color_control.setStreamName('color_control')
        self.color_control.out.link(self.color.inputControl)

        self.imu: dai.node.IMU = add_imu_node(pipeline)

class SetupColorYolo(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution, square: bool, mount: WarpConfig, nn_path: Path) -> None:
        super().__init__(pipeline, fps, resolution, square, mount)

        self.detection_manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.detection_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if square:
            self.detection_manip.initialConfig.setResize(*DETECTOR_INPUT_SQUARE)
            self.detection_manip.initialConfig.setKeepAspectRatio(True)
        else:
            self.detection_manip.initialConfig.setResize(*DETECTOR_INPUT_WIDE)
            self.detection_manip.initialConfig.setKeepAspectRatio(False)
        self.color_warp.out.link(self.detection_manip.inputImage)

        self.detection_network: dai.node.YoloDetectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.input.setBlocking(False)
        self.detection_manip.out.link(self.detection_network.input)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.outputTracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.outputTracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.outputTracklets.input)


class SetupColorStereo(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution, show_stereo:bool, lowres: bool = False) -> None:
        logger.warning("Color Stereo not implemented")
        return
        super().__init__(pipeline, fps, resolution, square = False)
        self.show_stereo: bool = show_stereo

        pipeline.remove(self.output_video)

        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)

        # The depth pair runs at the installation's mono mode, or a low mode when the depth
        # output is only feeding the detector (`lowres`).
        sensor_mode: dai.MonoCameraProperties.SensorResolution = mono_mode(resolution)
        if lowres:
            sensor_mode = dai.MonoCameraProperties.SensorResolution.THE_400_P

        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.left.setCamera("left")
        self.left.setResolution(sensor_mode)
        self.left.setFps(fps)

        self.right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.right.setCamera("right")
        self.right.setResolution(sensor_mode)
        self.right.setFps(fps)

        self.stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)
        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)

        self.sync: dai.node.Sync = pipeline.create(dai.node.Sync)
        sync_threshold = timedelta(seconds=(1.0 / self.fps) * 0.5)
        self.sync.setSyncAttempts(-1)
        self.sync.setSyncThreshold(sync_threshold)

        self.color.video.link(self.sync.inputs["video"])
        self.left.out.link(self.sync.inputs["left"])
        self.right.out.link(self.sync.inputs["right"])
        if self.show_stereo:
            self.stereo.disparity.link(self.sync.inputs["stereo"])

        self.output_sync: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_sync.setStreamName("sync")
        self.sync.out.link(self.output_sync.input)

        self.mono_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.mono_control.setStreamName('mono_control')
        self.mono_control.out.link(self.left.inputControl)
        self.mono_control.out.link(self.right.inputControl)

        self.stereo_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.stereo_control.setStreamName('stereo_control')
        self.stereo_control.out.link(self.stereo.inputConfig)

class SetupColorStereoYolo(SetupColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution, show_stereo: bool, nn_path: Path) -> None:
        logger.warning("Color Stereo not implemented")
        return
        super().__init__(pipeline, fps, resolution, show_stereo, lowres = True)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(*DETECTOR_INPUT_WIDE)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.color.video.link(self.manip.inputImage)

        self.detection_network: dai.node.YoloSpatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setSpatialCalculationAlgorithm(DEPTH_TRACKER_LOCATION)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)


class SetupMono(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 square: bool, mount: WarpConfig) -> None:
        super().__init__(pipeline, fps)


        self.resolution: CameraResolution = resolution
        self.sensor_mode: dai.MonoCameraProperties.SensorResolution = mono_mode(resolution)
        self.width, self.height = mono_frame_size(resolution)
        self.data_size = self.width * self.height

        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.left.setCamera("left")
        self.left.setResolution(self.sensor_mode)
        self.left.setFps(self.fps)

        self.left_warp: dai.node.Warp = pipeline.create(dai.node.Warp)
        self.mode_width: int = self.width   # `fov_h` is quoted for the un-cropped width
        self.mount: WarpConfig = mount

        if square:
            self.width = self.height        # make square
            self.data_size = self.width * self.height   # follow the crop, as SetupColor does

        warp_mesh, mesh_w, mesh_h = build_warp_mesh(
            (self.mode_width, self.height), (self.width, self.height), self.mode_width, mount)

        self.left_warp.setMaxOutputFrameSize(self.data_size)
        self.left_warp.setOutputSize(self.width, self.height)

        self.left_warp.setWarpMesh(warp_mesh, mesh_w, mesh_h)
        self.left.out.link(self.left_warp.inputImage)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.left_warp.out.link(self.output_video.input)

        self.mono_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.mono_control.setStreamName('mono_control')
        self.mono_control.out.link(self.left.inputControl)

        self.imu: dai.node.IMU = add_imu_node(pipeline)

class SetupMonoYolo(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 square: bool, mount: WarpConfig, nn_path: Path) -> None:
        super().__init__(pipeline, fps, resolution, square, mount)

        self.detection_manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.detection_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if square:
            self.detection_manip.initialConfig.setResize(*DETECTOR_INPUT_SQUARE)
            self.detection_manip.initialConfig.setKeepAspectRatio(True)
        else:
            self.detection_manip.initialConfig.setResize(*DETECTOR_INPUT_WIDE)
            self.detection_manip.initialConfig.setKeepAspectRatio(False)
        self.left_warp.out.link(self.detection_manip.inputImage)

        self.detection_network: dai.node.YoloDetectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.input.setBlocking(False)
        self.detection_manip.out.link(self.detection_network.input)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_manip.out.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)


class SetupMonoStereo(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 show_stereo: bool) -> None:
        logger.warning("Mono Stereo not implemented")
        return
        super().__init__(pipeline, fps, resolution, square = False)
        self.show_stereo: bool = show_stereo
        pipeline.remove(self.output_video)

        self.right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.right.setCamera("right")
        self.right.setResolution(self.resolution)
        self.right.setFps(fps)

        self.stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)

        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)

        self.sync: dai.node.Sync = pipeline.create(dai.node.Sync)
        sync_threshold = timedelta(seconds=(1.0 / self.fps) * 0.5)
        self.sync.setSyncAttempts(-1)
        self.sync.setSyncThreshold(sync_threshold)

        self.stereo.rectifiedLeft.link(self.sync.inputs["video"])
        self.left.out.link(self.sync.inputs["left"])
        self.right.out.link(self.sync.inputs["right"])
        if self.show_stereo:
            self.stereo.disparity.link(self.sync.inputs["stereo"])

        self.output_sync: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_sync.setStreamName("sync")
        self.sync.out.link(self.output_sync.input)

        self.mono_control.out.link(self.right.inputControl)
        self.stereo_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.stereo_control.setStreamName('stereo_control')
        self.stereo_control.out.link(self.stereo.inputConfig)

class SetupMonoStereoYolo(SetupMonoStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 show_stereo: bool, nn_path: Path) -> None:
        logger.warning("Mono Stereo not implemented")
        return
        super().__init__(pipeline, fps, show_stereo)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(*DETECTOR_INPUT_WIDE)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.stereo.rectifiedLeft.link(self.manip.inputImage)

        self.detection_network: dai.node.YoloSpatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setSpatialCalculationAlgorithm(DEPTH_TRACKER_LOCATION)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)


class SimulationColor(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution, square: bool, mount: WarpConfig,
                 warp_clips: bool = False) -> None:
        super().__init__(pipeline, fps, resolution, square, mount)

        pipeline.remove(self.color)
        # Simulation still runs on a real device, so its IMU would report the orientation
        # of a box on a bench — nothing to do with the recording. Better absent than wrong.
        pipeline.remove(self.imu)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(self.data_size)

        if warp_clips:
            self.color_warp.setWarpMesh(*clip_equirect_mesh((self.width, self.height), self.mode_width, mount))
            self.ex_video.out.link(self.color_warp.inputImage)
        else:
            pipeline.remove(self.color_warp)
            self.ex_video.out.link(self.output_video.input)

class SimulationColorYolo(SetupColorYolo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution, square: bool, mount: WarpConfig,
                 nn_path: Path, warp_clips: bool = False) -> None:
        super().__init__(pipeline, fps, resolution, square, mount, nn_path)

        pipeline.remove(self.color)
        # Simulation still runs on a real device, so its IMU would report the orientation
        # of a box on a bench — nothing to do with the recording. Better absent than wrong.
        pipeline.remove(self.imu)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(self.data_size)

        if warp_clips:
            self.color_warp.setWarpMesh(*clip_equirect_mesh((self.width, self.height), self.mode_width, mount))
            self.ex_video.out.link(self.color_warp.inputImage)
        else:
            pipeline.remove(self.color_warp)
            self.ex_video.out.link(self.detection_manip.inputImage)
            self.ex_video.out.link(self.output_video.input)
        self.ex_video.out.link(self.output_video.input)

class SimulationColorStereo(SetupColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution, show_stereo: bool) -> None:
        logger.warning("Color Stereo not implemented")
        return
        super().__init__(pipeline, fps, show_stereo)

        pipeline.remove(self.left)
        pipeline.remove(self.right)
        pipeline.remove(self.sync)
        pipeline.remove(self.output_sync)
        pipeline.remove(self.output_video)
        pipeline.remove(self.color_control)
        pipeline.remove(self.mono_control)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(self.data_size)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_left")
        self.ex_left.setMaxDataSize(self.data_size)

        self.ex_right: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_right.setStreamName("ex_right")
        self.ex_right.setMaxDataSize(self.data_size)

        self.ex_left.out.link(self.stereo.left)
        self.ex_right.out.link(self.stereo.right)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.ex_video.out.link(self.output_video.input)

        self.output_left: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_left.setStreamName("left")
        self.ex_left.out.link(self.output_left.input)
        # self.stereo.syncedLeft.link(self.output_left.input)

        self.output_right: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_right.setStreamName("right")
        self.ex_right.out.link(self.output_right.input)
        # self.stereo.syncedRight.link(self.output_right.input)

        if self.show_stereo:
            self.output_stereo: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
            self.output_stereo.setStreamName("stereo")
            self.stereo.disparity.link(self.output_stereo.input)

class SimulationColorStereoYolo(SimulationColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: float,  resolution: CameraResolution, show_stereo: bool, nn_path: Path) -> None:
        logger.warning("Color Stereo not implemented")
        return
        super().__init__(pipeline, fps, resolution, show_stereo)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(*DETECTOR_INPUT_WIDE)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.ex_video.out.link(self.manip.inputImage)

        self.detection_network: dai.node.YoloSpatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setSpatialCalculationAlgorithm(DEPTH_TRACKER_LOCATION)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.trackerOut: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.trackerOut.setStreamName("tracklets")
        self.object_tracker.out.link(self.trackerOut.input)

        pipeline.remove(self.output_left)
        pipeline.remove(self.output_right)


class SimulationMono(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 square: bool, mount: WarpConfig, warp_clips: bool = False) -> None:
        super().__init__(pipeline, fps, resolution, square, mount)

        pipeline.remove(self.left)
        pipeline.remove(self.mono_control)
        # Simulation still runs on a real device, so its IMU would report the orientation
        # of a box on a bench — nothing to do with the recording. Better absent than wrong.
        pipeline.remove(self.imu)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_video")
        self.ex_left.setMaxDataSize(self.data_size)

        if warp_clips:
            self.left_warp.setWarpMesh(*clip_equirect_mesh((self.width, self.height), self.mode_width, mount))
            self.ex_left.out.link(self.left_warp.inputImage)
        else:
            self.ex_left.out.link(self.output_video.input)

class SimulationMonoYolo(SetupMonoYolo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 square: bool, mount: WarpConfig, nn_path: Path, warp_clips: bool = False) -> None:
        super().__init__(pipeline, fps, resolution, square, mount, nn_path)

        pipeline.remove(self.left)
        pipeline.remove(self.mono_control)
        # Simulation still runs on a real device, so its IMU would report the orientation
        # of a box on a bench — nothing to do with the recording. Better absent than wrong.
        pipeline.remove(self.imu)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_video")
        self.ex_left.setMaxDataSize(self.data_size)

        if warp_clips:
            self.left_warp.setWarpMesh(*clip_equirect_mesh((self.width, self.height), self.mode_width, mount))
            self.ex_left.out.link(self.left_warp.inputImage)
        else:
            self.ex_left.out.link(self.detection_manip.inputImage)
            self.ex_left.out.link(self.output_video.input)
        # self.ex_left.out.link(self.output_video.input)

class SimulationMonoStereo(SetupMonoStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 show_stereo: bool) -> None:
        logger.warning("Mono Stereo not implemented")
        return
        super().__init__(pipeline, fps, show_stereo)

        self.color: dai.node.Camera = pipeline.create(dai.node.Camera)
        self.color.setCamera("color")
        # (unreachable — the early return above; a colour node in a mono setup, so it needs a
        #  colour size, not this class's mono `self.resolution`)
        self.color.setSize(*color_frame_size(CameraResolution.P720))
        self.color.setFps(self.fps)
        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)

        pipeline.remove(self.left)
        pipeline.remove(self.right)
        pipeline.remove(self.sync)
        pipeline.remove(self.output_video)
        pipeline.remove(self.output_sync)
        pipeline.remove(self.mono_control)

        self.stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(self.data_size)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_left")
        self.ex_left.setMaxDataSize(self.data_size)

        self.ex_right: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_right.setStreamName("ex_right")
        self.ex_right.setMaxDataSize(self.data_size)

        self.ex_left.out.link(self.stereo.left)
        self.ex_right.out.link(self.stereo.right)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.ex_video.out.link(self.output_video.input)

        self.output_left: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_left.setStreamName("left")
        self.ex_left.out.link(self.output_left.input)

        self.output_right: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_right.setStreamName("right")
        self.ex_right.out.link(self.output_right.input)

        if self.show_stereo:
            self.output_stereo: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
            self.output_stereo.setStreamName("stereo")
            self.stereo.disparity.link(self.output_stereo.input)

class SimulationMonoStereoYolo(SimulationMonoStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, resolution: CameraResolution,
                 show_stereo: bool, nn_path: Path) -> None:
        logger.warning("Mono Stereo not implemented")
        return
        super().__init__(pipeline, fps, show_stereo)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(*DETECTOR_INPUT_WIDE)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.ex_video.out.link(self.manip.inputImage)

        self.detection_network: dai.node.YoloSpatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setSpatialCalculationAlgorithm(DEPTH_TRACKER_LOCATION)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)

        pipeline.remove(self.output_left)
        pipeline.remove(self.output_right)

def find_equirect_warp(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: WarpConfig,
    mesh_w: int = WARP_MESH,
    mesh_h: int = WARP_MESH,
) -> list[dai.Point2f]:
    """The Warp node's mesh for undoing `mount.tilt`, as depthai points.

    All of the geometry lives in `definitions.equirect_mesh_points`, which has no depthai or
    OpenCV dependency and carries the explanation of the lens model and the mesh density.
    This is only the adapter. For the *other* correction, `keystone`, see `find_keystone_warp`.
    """
    return [dai.Point2f(float(x), float(y)) for x, y in equirect_mesh_points(
        src_size, out_size, mode_width, mount.fov_h, mount.tilt,
        mount.flip_h, mount.flip_v, mesh_w, mesh_h)]


def clip_equirect_mesh(
    frame_size: tuple[int, int],
    mode_width: int,
    mount: WarpConfig,
) -> tuple[list[dai.Point2f], int, int]:
    """The warp mesh for a RECORDING rather than a sensor, ready to splat into `setWarpMesh`.

    Recordings come off `output_video`, which is **after** the warp, so a clip already carries
    the flips — and the square crop — that were applied when it was shot. Re-applying the live
    mesh would do all of that a second time, un-mirroring the image and cropping an already
    cropped frame. What is wanted is only the *difference*, which is the tilt on its own:

    - the flips come off, because they are already in the pixels;
    - the frame is its own source, because the crop is already in the pixels;
    - the tilt's sign flips under `flip_v`, because a vertical mirror reverses a rotation about
      the horizontal axis, while a horizontal mirror commutes with it and needs no correction.

    ASSUMES THE CLIP IS RAW: equidistant, shot at `tilt = 0`, before the warp reprojected its
    output. That holds for every recording made before `tilt` existed. A clip recorded since
    is already equirectangular and would be reprojected twice; capture-time geometry is still
    not stored beside clips — see the plan's "save the preset alongside each recording".
    """
    clip_mount = WarpConfig(flip_h=False, flip_v=False,
                            tilt=-mount.tilt if mount.flip_v else mount.tilt,
                            keystone=0.0,           # a recording is already post-keystone
                            fov_h=mount.fov_h)
    mesh: list[dai.Point2f] = find_equirect_warp(frame_size, frame_size, mode_width, clip_mount)
    return mesh, WARP_MESH, WARP_MESH


# ---------------------------------------------------------------------------
#  Keystone — the pre-existing correction, kept verbatim
# ---------------------------------------------------------------------------
# These two builders are the original `find_perspective_warp` / `_square`, unchanged apart
# from the name, at their original 2 x 64 mesh. They build a homography that pins all four
# corners of the output to the frame corners: a rotation composed with a stretch-to-fit. That
# is the wrong model for re-aiming a wide lens (see `definitions.equirect_mesh_points`), but it is
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
    equidistant -> equirectangular reprojection that makes a column one azimuth (see
    `definitions.equirect_mesh_points`). Only a non-zero `keystone` reaches the old builder.
    """
    if mount.tilt != 0.0 and mount.keystone != 0.0:
        logger.warning("tilt (%.1f) and keystone (%.2f) are both set; they are exclusive — using tilt",
                       mount.tilt, mount.keystone)
    if mount.tilt != 0.0 or mount.keystone == 0.0:
        return find_equirect_warp(src_size, out_size, mode_width, mount), WARP_MESH, WARP_MESH

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