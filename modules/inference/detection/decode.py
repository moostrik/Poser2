"""From the YOLOv8 output to person boxes: pure numpy, so it is tested without a model or a GPU.

The exported model gives one row per candidate, (4 + classes) x candidates: centre x, centre y,
width and height in input pixels, then a score per COCO class with the sigmoid applied. Person is
class 0.
"""

import numpy as np

PERSON: int = 0


def person_boxes(output: np.ndarray, confidence: float, overlap: float) -> tuple[np.ndarray, np.ndarray]:
    """The person boxes of one image's output, (4 + classes, N), after the score threshold and the
    non-maximum suppression: boxes (M, 4) as x0, y0, x1, y1 in input pixels, scores (M,), both
    sorted by score, highest first."""
    scores: np.ndarray = output[4 + PERSON]
    keep: np.ndarray = scores >= confidence
    if not keep.any():
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    cx, cy, w, h = output[0, keep], output[1, keep], output[2, keep], output[3, keep]
    boxes: np.ndarray = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
    scores = scores[keep].astype(np.float32)
    order: np.ndarray = nms(boxes, scores, overlap)
    return boxes[order], scores[order]


def nms(boxes: np.ndarray, scores: np.ndarray, overlap: float) -> np.ndarray:
    """Greedy non-maximum suppression: the indices to keep, by score, highest first. A box is
    dropped when it overlaps a kept box by more than `overlap` (intersection over union)."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas: np.ndarray = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    order: np.ndarray = np.argsort(-scores, kind='stable')
    kept: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        kept.append(i)
        rest: np.ndarray = order[1:]
        if len(rest) == 0:
            break
        iw: np.ndarray = np.maximum(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0)
        ih: np.ndarray = np.maximum(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0)
        inter: np.ndarray = iw * ih
        union: np.ndarray = areas[i] + areas[rest] - inter
        iou: np.ndarray = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= overlap]
    return np.asarray(kept, dtype=np.int64)


def normalise(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Boxes in input pixels to x, y, width, height as fractions of the input, clipped to it. The
    input was the frame stretched whole into the model, so these are fractions of the frame too."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    x0: np.ndarray = np.clip(boxes[:, 0] / width, 0.0, 1.0)
    y0: np.ndarray = np.clip(boxes[:, 1] / height, 0.0, 1.0)
    x1: np.ndarray = np.clip(boxes[:, 2] / width, 0.0, 1.0)
    y1: np.ndarray = np.clip(boxes[:, 3] / height, 0.0, 1.0)
    return np.stack([x0, y0, x1 - x0, y1 - y0], axis=1).astype(np.float32)
