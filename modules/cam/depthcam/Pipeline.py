import depthai as dai
from datetime import timedelta
from pathlib import Path

from modules.cam.depthcam.Definitions import *
from dataclasses import dataclass

@dataclass
class PerspectiveConfig:
    flip_h: bool
    flip_v: bool
    perspective: float

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
    do_yolo: bool = True,
    perspective: PerspectiveConfig = PerspectiveConfig(False, False, 0.0),
    simulate: bool = False
    ) -> None:


    options: list[str] = [
        'Square,' if square else 'Wide,',
        'Color,' if do_color else 'Mono,',
        'Yolo,' if do_yolo else '',
        'Simulate' if simulate else ''
    ]

    pipeline_description = "Depth Pipeline: " + " ".join(filter(None, options))
    print(pipeline_description)

    nn_path: Path = get_model_path(model_path, square, False, simulate)
    if not simulate:
        if do_color:
            if do_yolo:
                SetupColorYolo(pipeline, fps, square, nn_path)
            else:
                SetupColor(pipeline, fps, square)
        else:
            if do_yolo:
                SetupMonoYolo(pipeline, fps, square, nn_path, perspective)
            else:
                SetupMono(pipeline, fps, square, perspective)
    else:
        if do_color:
            if do_yolo:
                SimulationColorYolo(pipeline, fps, square, nn_path)
            else:
                SimulationColor(pipeline, fps, square)
        else:
            if do_yolo:
                SimulationMonoYolo(pipeline, fps, square, nn_path)
            else:
                SimulationMono(pipeline, fps, square)


class Setup():
    def __init__(self, pipeline : dai.Pipeline, fps: float) -> None:
        self.pipeline: dai.Pipeline = pipeline
        self.fps: float = fps

class SetupColor(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool, perspective: PerspectiveConfig = PerspectiveConfig(True, False, 0.280)) -> None:
        super().__init__(pipeline, fps)

        width, height = 1920, 1072 # for warping must be devisible by 16

        self.resolution: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        self.color: dai.node.ColorCamera = pipeline.create(dai.node.ColorCamera)
        self.color.setResolution(self.resolution)
        self.color.setFps(self.fps)
        self.color.setInterleaved(False)
        self.color.setPreviewSize(width, height)

        self.color_warp: dai.node.Warp = pipeline.create(dai.node.Warp)
        warp_p: float = width * 0.5 * perspective.perspective
        mesh_w, mesh_h = 2, 64

        if square:
            warp_p: float = height * 0.5 * perspective.perspective
            warp_mesh: list[dai.Point2f] = find_perspective_warp_square(width, height, height, warp_p, perspective.flip_h, perspective.flip_v, mesh_w, mesh_h)
            width = height # make square
        else:
            warp_mesh: list[dai.Point2f] = find_perspective_warp(width, height, warp_p, perspective.flip_h, perspective.flip_v, mesh_w, mesh_h)

        self.color_warp.setMaxOutputFrameSize(height * height * 3)
        self.color_warp.setOutputSize(height, height)

        self.color_warp.setWarpMesh(warp_mesh, mesh_w, mesh_h)
        self.color.preview.link(self.color_warp.inputImage)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.color_warp.out.link(self.output_video.input)

        self.color_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.color_control.setStreamName('color_control')
        self.color_control.out.link(self.color.inputControl)

class SetupColorYolo(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool, nn_path: Path) -> None:
        super().__init__(pipeline, fps, square)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if square:
            self.manip.initialConfig.setResize(416, 416)
            self.manip.initialConfig.setKeepAspectRatio(True)
        else:
            self.manip.initialConfig.setResize(640,352)
            self.manip.initialConfig.setKeepAspectRatio(False)
        self.color_warp.out.link(self.manip.inputImage)

        self.detection_network: dai.node.YoloDetectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)

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

class SetupMono(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool, perspective: PerspectiveConfig = PerspectiveConfig(False, False, 0.0)) -> None:
        super().__init__(pipeline, fps)

        width, height = 1280, 720 # for warping must be devisible by 16

        self.resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.left.setCamera("left")
        self.left.setResolution(self.resolution)
        self.left.setFps(self.fps)

        self.left_warp: dai.node.Warp = pipeline.create(dai.node.Warp)
        warp_p: float = 1280 * 0.5 * perspective.perspective
        mesh_w, mesh_h = 2, 64

        if square:
            warp_p: float = height * 0.5 * perspective.perspective
            warp_mesh: list[dai.Point2f] = find_perspective_warp_square(1280, 720, height, warp_p, perspective.flip_h, perspective.flip_v, mesh_w, mesh_h)
            width = height # make square
        else:
            warp_mesh: list[dai.Point2f] = find_perspective_warp(1280, 720, warp_p, perspective.flip_h, perspective.flip_v, mesh_w, mesh_h)

        self.left_warp.setMaxOutputFrameSize(height * height * 3)
        self.left_warp.setOutputSize(height, height)

        self.left_warp.setWarpMesh(warp_mesh, mesh_w, mesh_h)
        self.left.out.link(self.left_warp.inputImage)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.left_warp.out.link(self.output_video.input)

        self.mono_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.mono_control.setStreamName('mono_control')
        self.mono_control.out.link(self.left.inputControl)

class SetupMonoYolo(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool, nn_path: Path, perspective: PerspectiveConfig = PerspectiveConfig(False, False, 0.0)) -> None:
        super().__init__(pipeline, fps, square, perspective)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if square:
            self.manip.initialConfig.setResize(416, 416)
            self.manip.initialConfig.setKeepAspectRatio(True)
        else:
            self.manip.initialConfig.setResize(640,352)
            self.manip.initialConfig.setKeepAspectRatio(False)
        self.left_warp.out.link(self.manip.inputImage)

        self.detection_network: dai.node.YoloDetectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setNumClasses(80)
        self.detection_network.setCoordinateSize(4)
        self.detection_network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        self.detection_network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.manip.out.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)

class SimulationColor(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool) -> None:
        super().__init__(pipeline, fps, square)

        pipeline.remove(self.color)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(1280*720*3)

        self.ex_video.out.link(self.output_video.input)

class SimulationColorYolo(SetupColorYolo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool, nn_path: Path) -> None:
        super().__init__(pipeline, fps, square, nn_path)

        pipeline.remove(self.color)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(1280*720*3)
        self.ex_video.out.link(self.manip.inputImage)
        self.ex_video.out.link(self.output_video.input)

class SimulationMono(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool) -> None:
        super().__init__(pipeline, fps, square)

        pipeline.remove(self.left)
        pipeline.remove(self.mono_control)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_video")
        self.ex_left.setMaxDataSize(1280*720*3)

        self.ex_left.out.link(self.output_video.input)

class SimulationMonoYolo(SetupMonoYolo):
    def __init__(self, pipeline : dai.Pipeline, fps: float, square: bool, nn_path: Path) -> None:
        super().__init__(pipeline, fps, square, nn_path)

        pipeline.remove(self.left)
        pipeline.remove(self.mono_control)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_video")
        self.ex_left.setMaxDataSize(1280*720*3)

        self.ex_left.out.link(self.manip.inputImage)
        self.ex_left.out.link(self.output_video.input)

import cv2
import numpy as np

def find_perspective_warp(width, height, width_offset, flip_h, flip_v, mesh_w, mesh_h)  -> list[dai.Point2f]:
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

def find_perspective_warp_square(src_width, src_height, square_size, width_offset, flip_h, flip_v, mesh_w, mesh_h) -> list[dai.Point2f]:
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