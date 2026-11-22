"""The models on the OAK4 and its tracker: which chain runs, their archives, and their output as the contract
carries it."""

import logging
from enum import IntEnum

import depthai as dai
import numpy as np

logger = logging.getLogger(__name__)


class ModelChain(IntEnum):
    """The models on the camera; the archives are fetched by `scripts/oak4_models.py` into `data/models`."""
    OFF = 0             # no model on the camera: frames only
    DETECTOR = 1        # YOLOv8n: boxes only
    ONE_MODEL = 2       # YOLOv8-pose: boxes and keypoints in one pass
    TWO_MODELS = 3      # YOLOv6 detector, then Lite-HRNet on each player's crop


PERSON_DETECTOR_ARCHIVE: str = 'yolov8n_coco_640x512_rvc4.tar.xz'     # compiled through HubAI: the zoo has no 5:4 size
POSE_MODEL: str = 'luxonis/yolov8-nano-pose-estimation:coco-512x288'
POSE_ARCHIVE: str = 'yolov8n_pose_coco_512x288_rvc4.tar.xz'
DETECTOR_MODEL: str = 'luxonis/yolov6-nano:r2-coco-512x288'
DETECTOR_ARCHIVE: str = 'yolov6n_coco_512x288_rvc4.tar.xz'
TOP_DOWN_MODEL: str = 'luxonis/lite-hrnet:18-coco-192x256'
TOP_DOWN_ARCHIVE: str = 'lite_hrnet18_coco_192x256_rvc4.tar.xz'
NUM_KEYPOINTS: int = 17                 # COCO, the order of `modules.pose.PointLandmark`
POSE_CROP_PADDING: float = 0.0          # margin round the player's box before the crop, normalized to the frame
DETECTOR_CONFIDENCE_THRESHOLD: float = 0.5   # the first model of every chain, a pose model included
DETECTOR_OVERLAP_THRESHOLD: float = 0.5

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
    """The detection's landmarks as the contract carries them: (17, 3) float32, x and y normalized
    to the frame the detector saw (the delivered frame, stretched into the model), and the score;
    None when the detection carries none."""
    points = detection.getKeypoints()
    if len(points) != NUM_KEYPOINTS:
        return None
    return np.array([(p.imageCoordinates.x, p.imageCoordinates.y, p.confidence) for p in points], dtype=np.float32)


def keypoints_in_frame(message: dai.beta.Keypoints, frame: dai.ImgTransformation | None) -> np.ndarray | None:
    """A crop's keypoints from the top-down pose model, in the delivered frame, as the contract
    carries them: (17, 3) float32, x and y normalized to the frame, and the score. The message's
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
