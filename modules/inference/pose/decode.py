"""RTMPose's SimCC outputs decoded to keypoints and scores, as MMPose's `get_simcc_maximum`."""

import numpy as np

NUM_KEYPOINTS: int = 17         # COCO
SIMCC_SPLIT_RATIO: float = 2.0  # RTMPose's: the SimCC bins per input pixel


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = SIMCC_SPLIT_RATIO,
                 apply_softmax: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints (N, K, 2) in the model input's pixels and scores (N, K) in 0..1 from the SimCC outputs (N, K, W) and
    (N, K, H): each axis at its maximum, the score the lower of the two maxima; a keypoint without score is -1.

    Reference: mmpose/codecs/utils/post_processing.py::get_simcc_maximum
    """
    N, K, _ = simcc_x.shape

    simcc_x_flat = simcc_x.reshape(N * K, -1)
    simcc_y_flat = simcc_y.reshape(N * K, -1)

    if apply_softmax:
        simcc_x_flat = simcc_x_flat - np.max(simcc_x_flat, axis=1, keepdims=True)
        simcc_y_flat = simcc_y_flat - np.max(simcc_y_flat, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x_flat), np.exp(simcc_y_flat)
        simcc_x_flat = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y_flat = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x_flat, axis=1)
    y_locs = np.argmax(simcc_y_flat, axis=1)

    max_val_x = np.amax(simcc_x_flat, axis=1)
    max_val_y = np.amax(simcc_y_flat, axis=1)

    # MMPose takes the minimum of the x and y scores
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    scores = max_val_x

    x_coords = x_locs.astype(np.float32) / split_ratio
    y_coords = y_locs.astype(np.float32) / split_ratio

    keypoints = np.stack([x_coords, y_coords], axis=-1).reshape(N, K, 2)
    scores = scores.reshape(N, K)

    keypoints[scores <= 0.] = -1
    scores = np.clip(scores, 0.0, 1.0)
    return keypoints, scores
