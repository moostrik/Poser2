"""The 3D bridge's trail arithmetic without GL: the Ema weight, a trail's direction and its advection."""

import math

import numpy as np

EPSILON: float = 1e-4       # below this a speed is no motion and a trail has no direction; the same value in the shaders


def decay_weight(dt: float, time: float) -> float:
    """What remains of a value after `dt` seconds with time constant `time` in ms: exp(-dt / time); 0 ms forgets at once."""
    if time <= 0.0:
        return 0.0
    return math.exp(-dt * 1000.0 / time)


def advect(field: np.ndarray, displacement: np.ndarray, amount: float) -> np.ndarray:
    """`field` (H, W, C) moved along `displacement` (H, W, 2, uv per frame) by `amount`: each pixel takes the
    value from where it came from, linear between pixels, clamped at the edges. Mirrors trail_advect.frag."""
    h, w = field.shape[:2]
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    x: np.ndarray = np.clip(columns - amount * displacement[..., 0] * w, 0.0, w - 1.0)
    y: np.ndarray = np.clip(rows - amount * displacement[..., 1] * h, 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top: np.ndarray = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
    bottom: np.ndarray = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def direction(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """Unit direction of the motion, per pixel (..., C): the fast trail's, the slow one's when the fast trail has stopped."""
    fast_length: np.ndarray = np.linalg.norm(fast, axis=-1, keepdims=True)
    slow_length: np.ndarray = np.linalg.norm(slow, axis=-1, keepdims=True)
    fast_direction: np.ndarray = np.divide(fast, fast_length, out=np.zeros_like(fast), where=fast_length > EPSILON)
    slow_direction: np.ndarray = np.divide(slow, slow_length, out=np.zeros_like(slow), where=slow_length > EPSILON)
    return np.where(fast_length > EPSILON, fast_direction, slow_direction)
