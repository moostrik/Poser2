"""SCRFD's outputs to faces (insightface antelopev2 'scrfd_10g_bnkps', the model InstantID uses): per stride the
scores, the box distances and the keypoint distances from each anchor's center, decoded and kept above a threshold,
then non-maximum suppression. Follows insightface/model_zoo/scrfd.py, without the insightface package."""

import cv2
import numpy as np

INPUT_SIZE: int = 640
FEAT_STRIDES: tuple[int, ...] = (8, 16, 32)
NUM_ANCHORS: int = 2
NMS_THRESHOLD: float = 0.4

_anchor_centers: dict[int, np.ndarray] = {}


def anchor_centers(stride: int) -> np.ndarray:
    """The anchors' centers of one stride on the 640 input, (cells x anchors, 2)."""
    if stride not in _anchor_centers:
        size: int = INPUT_SIZE // stride
        centers: np.ndarray = np.stack(list(np.mgrid[:size, :size][::-1]), axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        _anchor_centers[stride] = np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
    return _anchor_centers[stride]


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x0 = points[:, 0] - distance[:, 0]
    y0 = points[:, 1] - distance[:, 1]
    x1 = points[:, 0] + distance[:, 2]
    y1 = points[:, 1] + distance[:, 3]
    return np.stack([x0, y0, x1, y1], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds: list[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1).reshape((distance.shape[0], -1, 2))


def decode(outputs: list[np.ndarray], scale: float, threshold: float) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """The faces of one image from SCRFD's nine outputs (scores, boxes, keypoints per stride, the batch of one
    dropped), in the image's pixels (`scale` the input's size over the image's), as (box, score, keypoints),
    highest score first."""
    count: int = len(FEAT_STRIDES)
    scores_list, boxes_list, kps_list = [], [], []
    for index, stride in enumerate(FEAT_STRIDES):
        scores: np.ndarray = outputs[index].reshape(-1)
        box_distances: np.ndarray = outputs[index + count].reshape(-1, 4) * stride
        kps_distances: np.ndarray = outputs[index + count * 2].reshape(-1, 10) * stride
        centers: np.ndarray = anchor_centers(stride)
        kept: np.ndarray = np.where(scores >= threshold)[0]
        scores_list.append(scores[kept])
        boxes_list.append(distance2bbox(centers, box_distances)[kept])
        kps_list.append(distance2kps(centers, kps_distances)[kept])
    all_scores: np.ndarray = np.concatenate(scores_list)
    if len(all_scores) == 0:
        return []
    all_boxes: np.ndarray = np.concatenate(boxes_list) / scale
    all_kps: np.ndarray = np.concatenate(kps_list) / scale
    rects: list = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in all_boxes]
    keep = cv2.dnn.NMSBoxes(rects, all_scores.tolist(), threshold, NMS_THRESHOLD)
    faces = [(all_boxes[i], float(all_scores[i]), all_kps[i]) for i in np.array(keep).reshape(-1)]
    faces.sort(key=lambda face: face[1], reverse=True)
    return faces
