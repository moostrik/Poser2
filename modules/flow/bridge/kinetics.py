"""The kinetic bridge's arithmetic without GL: the trail weight, and a numpy reference of its shaders."""

import math

import numpy as np

EPSILON: float = 1e-4       # below this a trail has no direction; the same value in the kinetic shaders


def decay_weight(dt: float, time: float) -> float:
    """What remains of a value after `dt` seconds with time constant `time` in ms: exp(-dt / time); 0 ms forgets at once."""
    if time <= 0.0:
        return 0.0
    return math.exp(-dt * 1000.0 / time)


def smooth_depth(depth: np.ndarray, previous: np.ndarray, weight: float, jump: float) -> np.ndarray:
    """Depth in metres through an Ema, 0 where there is none; a pixel that jumps by `jump` restarts from the new depth."""
    tracked: np.ndarray = (depth > 0.0) & (previous > 0.0) & (np.abs(depth - previous) <= jump)
    return np.where(tracked, previous * weight + depth * (1.0 - weight), depth)


def depth_speed(smoothed: np.ndarray, previous: np.ndarray, dt: float, jump: float) -> np.ndarray:
    """Velocity toward the camera in m/s; 0 without depth at both frames, and at a jump."""
    change: np.ndarray = smoothed - previous
    moving: np.ndarray = (smoothed > 0.0) & (previous > 0.0) & (np.abs(change) <= jump)
    return np.where(moving, -change / dt, 0.0)


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


def hold_release(acceleration: np.ndarray, held: np.ndarray, decay: float) -> np.ndarray:
    """The peak-held release: the new release, or what remains of the last one when that is more."""
    return np.maximum(np.maximum(0.0, -acceleration), held * decay)


def drop(fast: np.ndarray, slow: np.ndarray, acceleration: np.ndarray, amount: float) -> np.ndarray:
    """The velocity output: the fast trail, blended to the slow one by `amount` * the release relative to the slow speed."""
    release: np.ndarray = np.maximum(0.0, -acceleration)
    weight: np.ndarray = np.clip(release / np.maximum(np.linalg.norm(slow, axis=-1), EPSILON), 0.0, 1.0)
    return fast + (slow - fast) * (amount * weight)[..., None]


def fill_reservoir(reservoir: np.ndarray, speed: np.ndarray, held_release: np.ndarray, leak: float, dt: float, gain: float) -> np.ndarray:
    """The next reservoir: motion fills it, it leaks, and what a release emits leaves it."""
    filled: np.ndarray = reservoir * leak + speed * dt
    return filled * np.maximum(0.0, 1.0 - held_release * gain * dt)
