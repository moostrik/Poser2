"""The kinetic bridge's arithmetic without GL: the trail weight, and a numpy reference of its shaders."""

import math

import numpy as np

EPSILON: float = 1e-4       # below this a trail has no direction; the same value in the kinetic shaders


def decay_weight(dt: float, time: float) -> float:
    """What remains of a value after `dt` seconds with time constant `time` in ms: exp(-dt / time); 0 ms forgets at once."""
    if time <= 0.0:
        return 0.0
    return math.exp(-dt * 1000.0 / time)


def direction(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """Unit direction of the motion, per pixel (..., 2): the fast trail's, the slow one's when the fast trail has stopped."""
    fast_length: np.ndarray = np.linalg.norm(fast, axis=-1, keepdims=True)
    slow_length: np.ndarray = np.linalg.norm(slow, axis=-1, keepdims=True)
    fast_direction: np.ndarray = np.divide(fast, fast_length, out=np.zeros_like(fast), where=fast_length > EPSILON)
    slow_direction: np.ndarray = np.divide(slow, slow_length, out=np.zeros_like(slow), where=slow_length > EPSILON)
    return np.where(fast_length > EPSILON, fast_direction, slow_direction)


def project(fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Acceleration (along the motion, + onset, - release) and turn (across it, signed 2D cross) of fast - slow."""
    d: np.ndarray = fast - slow
    unit: np.ndarray = direction(fast, slow)
    acceleration: np.ndarray = np.sum(d * unit, axis=-1)
    turn: np.ndarray = unit[..., 0] * d[..., 1] - unit[..., 1] * d[..., 0]
    return acceleration, turn
