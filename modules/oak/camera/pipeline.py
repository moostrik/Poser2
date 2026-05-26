import depthai as dai
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

from .definitions import (
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


def build_pipeline(pipeline: dai.Pipeline, config: PipelineConfig) -> PipelineHandles:
    square = config.square
    mesh_w, mesh_h = 2, 64

    # Determine source dimensions
    if config.do_color:
        if config.do_720p:
            width, height = 1280, 720
            resolution = dai.ColorCameraProperties.SensorResolution.THE_720_P
        else:
            width, height = 1920, 1072
            resolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
    else:
        width, height = 1280, 720

    # Warp output dimensions
    if square:
        warp_p = height * 0.5 * config.perspective.perspective
        warp_mesh = find_perspective_warp_square(
            width, height, height, warp_p,
            config.perspective.flip_h, config.perspective.flip_v,
            mesh_w, mesh_h,
        )
        out_w = out_h = height
    else:
        warp_p = width * 0.5 * config.perspective.perspective
        warp_mesh = find_perspective_warp(
            width, height, warp_p,
            config.perspective.flip_h, config.perspective.flip_v,
            mesh_w, mesh_h,
        )
        out_w, out_h = width, height

    data_size = out_w * out_h * (3 if config.do_color else 1)

    options: list[str] = [
        'Square,' if square else 'Wide,',
        'Color,' if config.do_color else 'Mono,',
        'Yolo,' if config.do_yolo else '',
        'Simulate' if config.simulate else '',
    ]
    logger.info("Pipeline: " + " ".join(filter(None, options)))

    if config.simulate:
        # XLinkIn replaces camera + warp
        ex_video = pipeline.create(dai.node.XLinkIn)
        ex_video.setStreamName("ex_video")
        ex_video.setMaxDataSize(data_size)
        frame_source = ex_video.out
    else:
        if config.do_color:
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setResolution(resolution)
            cam.setFps(config.fps)
            cam.setInterleaved(False)
            cam.setPreviewSize(width, height)
            cam_out = cam.preview

            ctrl = pipeline.create(dai.node.XLinkIn)
            ctrl.setStreamName("color_control")
            ctrl.out.link(cam.inputControl)
        else:
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setCamera("left")
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            cam.setFps(config.fps)
            cam_out = cam.out

            ctrl = pipeline.create(dai.node.XLinkIn)
            ctrl.setStreamName("mono_control")
            ctrl.out.link(cam.inputControl)

        warp = pipeline.create(dai.node.Warp)
        warp.setMaxOutputFrameSize(data_size)
        warp.setOutputSize(out_w, out_h)
        warp.setWarpMesh(warp_mesh, mesh_w, mesh_h)
        cam_out.link(warp.inputImage)
        frame_source = warp.out

    output_video = pipeline.create(dai.node.XLinkOut)
    output_video.setStreamName("video")
    frame_source.link(output_video.input)

    if config.do_yolo:
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if square:
            manip.initialConfig.setResize(416, 416)
            manip.initialConfig.setKeepAspectRatio(True)
        else:
            manip.initialConfig.setResize(640, 352)
            manip.initialConfig.setKeepAspectRatio(False)
        frame_source.link(manip.inputImage)

        network = pipeline.create(dai.node.YoloDetectionNetwork)
        assert config.nn_path is not None
        network.setBlobPath(config.nn_path)
        network.setNumInferenceThreads(2)
        network.setNumClasses(80)
        network.setCoordinateSize(4)
        network.setConfidenceThreshold(YOLO_CONFIDENCE_THRESHOLD)
        network.setIouThreshold(YOLO_OVERLAP_THRESHOLD)
        network.input.setBlocking(False)
        manip.out.link(network.input)

        tracker = pipeline.create(dai.node.ObjectTracker)
        tracker.setDetectionLabelsToTrack([TRACKER_PERSON_LABEL])
        tracker.setTrackerType(TRACKER_TYPE)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        network.passthrough.link(tracker.inputTrackerFrame)
        network.passthrough.link(tracker.inputDetectionFrame)
        network.out.link(tracker.inputDetections)

        output_tracklets = pipeline.create(dai.node.XLinkOut)
        output_tracklets.setStreamName("tracklets")
        tracker.out.link(output_tracklets.input)

    return PipelineHandles(do_color=config.do_color, do_yolo=config.do_yolo)
