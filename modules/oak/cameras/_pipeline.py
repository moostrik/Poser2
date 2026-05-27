import depthai as dai
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

from ._definitions import (
    YOLOV8_SQUARE_6S, YOLOV8_SQUARE_7S, YOLOV8_WIDE_6S, YOLOV8_WIDE_7S,
    YOLO_CONFIDENCE_THRESHOLD, YOLO_OVERLAP_THRESHOLD,
    TRACKER_PERSON_LABEL, TRACKER_TYPE,
)

import cv2
import numpy as np




@dataclass
class PerspectiveConfig:
    flip_h: bool
    flip_v: bool
    perspective: float


def find_perspective_warp(width, height, width_offset, flip_h, flip_v, mesh_w, mesh_h) -> list[dai.Point2f]:
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
        [x_offset, y_offset],                              # Top-left
        [x_offset + square_size, y_offset],                # Top-right
        [x_offset + square_size, y_offset + square_size],  # Bottom-right
        [x_offset, y_offset + square_size]                 # Bottom-left
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


def get_model_path(model_path: str, square: bool, simulate: bool) -> Path:
    if square:
        blob = YOLOV8_SQUARE_7S if simulate else YOLOV8_SQUARE_6S
    else:
        blob = YOLOV8_WIDE_7S if simulate else YOLOV8_WIDE_6S
    return (Path(model_path) / blob).resolve().absolute()


@dataclass
class PipelineConfig:
    fps: float
    square: bool
    do_color: bool
    do_yolo: bool
    do_720p: bool
    perspective: PerspectiveConfig
    simulate: bool
    nn_path: Path | None  # required when do_yolo is True


@dataclass
class PipelineHandles:
    do_color: bool
    do_yolo: bool
    video_out: dai.Node.Output
    control_in: dai.Node.Input | None        # camera control input (None when simulate)
    tracklets_out: dai.Node.Output | None    # tracker output (when do_yolo)
    video_frame_in: dai.Node.Input | None    # frame injection point (when simulate)


def build_pipeline(pipeline: dai.Pipeline, config: PipelineConfig) -> PipelineHandles:
    options: list[str] = [
        'Square,' if config.square else 'Wide,',
        'Color,' if config.do_color else 'Mono,',
        'Yolo,' if config.do_yolo else '',
        'Simulate' if config.simulate else '',
    ]
    logger.info("Pipeline: " + " ".join(filter(None, options)))

    if config.simulate:
        if config.do_color:
            if config.do_yolo:
                s = _SimulateColorYolo(pipeline, config)
            else:
                s = _SimulateColor(pipeline, config)
        else:
            if config.do_yolo:
                s = _SimulateMonoYolo(pipeline, config)
            else:
                s = _SimulateMono(pipeline, config)
    else:
        if config.do_color:
            if config.do_yolo:
                s = _SetupColorYolo(pipeline, config)
            else:
                s = _SetupColor(pipeline, config)
        else:
            if config.do_yolo:
                s = _SetupMonoYolo(pipeline, config)
            else:
                s = _SetupMono(pipeline, config)

    return PipelineHandles(
        do_color=config.do_color,
        do_yolo=config.do_yolo,
        video_out=s.video_out,
        control_in=s.control_in,
        tracklets_out=s.tracklets_out,
        video_frame_in=s.video_frame_in,
    )


# ---------------------------------------------------------------------------
# Pipeline setup classes — one per use case
# ---------------------------------------------------------------------------

class _SetupColor:
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        mesh_w, mesh_h = 2, 64
        width, height = (1280, 720) if config.do_720p else (1920, 1072)
        if config.square:
            out_w = out_h = height
            warp_p = height * 0.5 * config.perspective.perspective
            warp_mesh = find_perspective_warp_square(width, height, height, warp_p, config.perspective.flip_h, config.perspective.flip_v, mesh_w, mesh_h)
        else:
            out_w, out_h = width, height
            warp_p = width * 0.5 * config.perspective.perspective
            warp_mesh = find_perspective_warp(width, height, warp_p, config.perspective.flip_h, config.perspective.flip_v, mesh_w, mesh_h)

        cam = pipeline.create(dai.node.Camera)
        cam.build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput((width, height), type=dai.ImgFrame.Type.BGR888p, fps=config.fps)

        warp = pipeline.create(dai.node.Warp)
        warp.setMaxOutputFrameSize(out_w * out_h * 3)
        warp.setOutputSize(out_w, out_h)
        warp.setWarpMesh([(p.x, p.y) for p in warp_mesh], mesh_w, mesh_h)
        cam_out.link(warp.inputImage)

        self._frame_source: dai.Node.Output = warp.out
        self.video_out: dai.Node.Output = warp.out
        self.control_in: dai.Node.Input | None = cam.inputControl
        self.tracklets_out: dai.Node.Output | None = None
        self.video_frame_in: dai.Node.Input | None = None


class _SetupColorYolo(_SetupColor):
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        super().__init__(pipeline, config)
        assert config.nn_path is not None
        yolo_w, yolo_h = (416, 416) if config.square else (640, 352)

        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(yolo_w, yolo_h)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self._frame_source.link(manip.inputImage)

        network = pipeline.create(dai.node.DetectionNetwork)
        network.setBlobPath(config.nn_path)
        network.setNumInferenceThreads(2)
        network.input.setBlocking(False)
        network.detectionParser.setNNFamily(dai.DetectionNetworkType.YOLO)
        network.detectionParser.setSubtype("yolov8n")
        network.detectionParser.setNumClasses(80)
        network.detectionParser.setCoordinateSize(4)
        network.detectionParser.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        network.detectionParser.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        network.detectionParser.setStrides([8, 16, 32])
        manip.out.link(network.input)

        tracker = pipeline.create(dai.node.ObjectTracker)
        tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        tracker.setTrackerType(TRACKER_TYPE)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        network.passthrough.link(tracker.inputTrackerFrame)
        network.passthrough.link(tracker.inputDetectionFrame)
        network.out.link(tracker.inputDetections)

        self.tracklets_out = tracker.out
        self.video_out = self._frame_source


class _SetupMono:
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        mesh_w, mesh_h = 2, 64
        width, height = 1280, 720
        if config.square:
            out_w = out_h = height
            warp_p = height * 0.5 * config.perspective.perspective
            warp_mesh = find_perspective_warp_square(width, height, height, warp_p, config.perspective.flip_h, config.perspective.flip_v, mesh_w, mesh_h)
        else:
            out_w, out_h = width, height
            warp_p = width * 0.5 * config.perspective.perspective
            warp_mesh = find_perspective_warp(width, height, warp_p, config.perspective.flip_h, config.perspective.flip_v, mesh_w, mesh_h)

        cam = pipeline.create(dai.node.Camera)
        cam.build(dai.CameraBoardSocket.CAM_B)
        cam_out = cam.requestOutput((width, height), type=dai.ImgFrame.Type.GRAY8, fps=config.fps)

        warp = pipeline.create(dai.node.Warp)
        warp.setMaxOutputFrameSize(out_w * out_h)
        warp.setOutputSize(out_w, out_h)
        warp.setWarpMesh([(p.x, p.y) for p in warp_mesh], mesh_w, mesh_h)
        cam_out.link(warp.inputImage)

        self._frame_source: dai.Node.Output = warp.out
        self.video_out: dai.Node.Output = warp.out
        self.control_in: dai.Node.Input | None = cam.inputControl
        self.tracklets_out: dai.Node.Output | None = None
        self.video_frame_in: dai.Node.Input | None = None


class _SetupMonoYolo(_SetupMono):
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        super().__init__(pipeline, config)
        assert config.nn_path is not None
        yolo_w, yolo_h = (416, 416) if config.square else (640, 352)

        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(yolo_w, yolo_h)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self._frame_source.link(manip.inputImage)

        network = pipeline.create(dai.node.DetectionNetwork)
        network.setBlobPath(config.nn_path)
        network.setNumInferenceThreads(2)
        network.input.setBlocking(False)
        network.detectionParser.setNNFamily(dai.DetectionNetworkType.YOLO)
        network.detectionParser.setSubtype("yolov8n")
        network.detectionParser.setNumClasses(80)
        network.detectionParser.setCoordinateSize(4)
        network.detectionParser.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        network.detectionParser.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        network.detectionParser.setStrides([8, 16, 32])
        manip.out.link(network.input)

        tracker = pipeline.create(dai.node.ObjectTracker)
        tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        tracker.setTrackerType(TRACKER_TYPE)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        network.passthrough.link(tracker.inputTrackerFrame)
        network.passthrough.link(tracker.inputDetectionFrame)
        network.out.link(tracker.inputDetections)

        self.tracklets_out = tracker.out
        self.video_out = self._frame_source


class _SimulateColor:
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        width, height = (1280, 720) if config.do_720p else (1920, 1072)
        out_w = out_h = height if config.square else None
        out_w = out_w or width
        out_h = out_h or height

        entry = pipeline.create(dai.node.ImageManip)
        entry.initialConfig.setOutputSize(out_w, out_h)
        entry.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        entry.setMaxOutputFrameSize(out_w * out_h * 3)

        self._frame_source: dai.Node.Output = entry.out
        self.video_out: dai.Node.Output = entry.out
        self.control_in: dai.Node.Input | None = None
        self.tracklets_out: dai.Node.Output | None = None
        self.video_frame_in: dai.Node.Input | None = entry.inputImage


class _SimulateColorYolo(_SimulateColor):
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        super().__init__(pipeline, config)
        assert config.nn_path is not None
        yolo_w, yolo_h = (416, 416) if config.square else (640, 352)

        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(yolo_w, yolo_h)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self._frame_source.link(manip.inputImage)

        network = pipeline.create(dai.node.DetectionNetwork)
        network.setBlobPath(config.nn_path)
        network.setNumInferenceThreads(2)
        network.input.setBlocking(False)
        network.detectionParser.setNNFamily(dai.DetectionNetworkType.YOLO)
        network.detectionParser.setSubtype("yolov8n")
        network.detectionParser.setNumClasses(80)
        network.detectionParser.setCoordinateSize(4)
        network.detectionParser.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        network.detectionParser.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        network.detectionParser.setStrides([8, 16, 32])
        manip.out.link(network.input)

        tracker = pipeline.create(dai.node.ObjectTracker)
        tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        tracker.setTrackerType(TRACKER_TYPE)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        network.passthrough.link(tracker.inputTrackerFrame)
        network.passthrough.link(tracker.inputDetectionFrame)
        network.out.link(tracker.inputDetections)

        self.tracklets_out = tracker.out
        self.video_out = self._frame_source


class _SimulateMono:
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        out_w = out_h = 720 if config.square else None
        out_w = out_w or 1280
        out_h = out_h or 720

        entry = pipeline.create(dai.node.ImageManip)
        entry.initialConfig.setOutputSize(out_w, out_h)
        entry.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        entry.setMaxOutputFrameSize(out_w * out_h)

        self._frame_source: dai.Node.Output = entry.out
        self.video_out: dai.Node.Output = entry.out
        self.control_in: dai.Node.Input | None = None
        self.tracklets_out: dai.Node.Output | None = None
        self.video_frame_in: dai.Node.Input | None = entry.inputImage


class _SimulateMonoYolo(_SimulateMono):
    def __init__(self, pipeline: dai.Pipeline, config: PipelineConfig) -> None:
        super().__init__(pipeline, config)
        assert config.nn_path is not None
        yolo_w, yolo_h = (416, 416) if config.square else (640, 352)

        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(yolo_w, yolo_h)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self._frame_source.link(manip.inputImage)

        network = pipeline.create(dai.node.DetectionNetwork)
        network.setBlobPath(config.nn_path)
        network.setNumInferenceThreads(2)
        network.input.setBlocking(False)
        network.detectionParser.setNNFamily(dai.DetectionNetworkType.YOLO)
        network.detectionParser.setSubtype("yolov8n")
        network.detectionParser.setNumClasses(80)
        network.detectionParser.setCoordinateSize(4)
        network.detectionParser.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        network.detectionParser.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        network.detectionParser.setStrides([8, 16, 32])
        manip.out.link(network.input)

        tracker = pipeline.create(dai.node.ObjectTracker)
        tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        tracker.setTrackerType(TRACKER_TYPE)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        network.passthrough.link(tracker.inputTrackerFrame)
        network.passthrough.link(tracker.inputDetectionFrame)
        network.out.link(tracker.inputDetections)

        self.tracklets_out = tracker.out
        self.video_out = self._frame_source

