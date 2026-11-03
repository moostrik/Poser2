"""The kinetic bridge's arithmetic without GL: the trail weight, and a numpy reference of its shaders."""

import math

import numpy as np

EPSILON: float = 1e-4       # below this a trail has no direction; the same value in the kinetic shaders


def decay_weight(dt: float, time: float) -> float:
    """What remains of a value after `dt` seconds with time constant `time` in ms: exp(-dt / time); 0 ms forgets at once."""
    if time <= 0.0:
        return 0.0
    return math.exp(-dt * 1000.0 / time)


def advect(field: np.ndarray, displacement: np.ndarray, amount: float) -> np.ndarray:
    """`field` (H, W, C) moved along `displacement` (H, W, 2, uv per frame) by `amount`: each pixel takes the
    value from where it came from, linear between pixels, clamped at the edges. Mirrors kinetic_advect.frag."""
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


def stop_gate(patch_fast: np.ndarray, patch_slow: np.ndarray, strength: float) -> np.ndarray:
    """How much of a release counts, from a patch's mean speeds: how far the patch is slowing down, 0..1.

    A limb moving steadily keeps its patch's mean speed, whatever it does to single pixels; a stop lowers it.
    Strength 0 lets every release through.
    """
    slowdown: np.ndarray = np.clip((patch_slow - patch_fast) / np.maximum(patch_slow, EPSILON), 0.0, 1.0)
    return 1.0 + (slowdown - 1.0) * strength


def hold_release(acceleration: np.ndarray, held: np.ndarray, decay: float, gate: np.ndarray | float = 1.0) -> np.ndarray:
    """The peak-held release: the new release through the gate, or what remains of the last one when that is more."""
    return np.maximum(np.maximum(0.0, -acceleration) * gate, held * decay)


def drop(fast: np.ndarray, slow: np.ndarray, acceleration: np.ndarray, amount: float) -> np.ndarray:
    """The velocity output: the fast trail, blended to the slow one by `amount` * the release relative to the slow speed."""
    release: np.ndarray = np.maximum(0.0, -acceleration)
    weight: np.ndarray = np.clip(release / np.maximum(np.linalg.norm(slow, axis=-1), EPSILON), 0.0, 1.0)
    return fast + (slow - fast) * (amount * weight)[..., None]


def fill_reservoir(reservoir: np.ndarray, speed: np.ndarray, held_release: np.ndarray, leak: float, dt: float, gain: float) -> np.ndarray:
    """The next reservoir: motion fills it, it leaks, and what a release emits leaves it."""
    filled: np.ndarray = reservoir * leak + speed * dt
    return filled * np.maximum(0.0, 1.0 - held_release * gain * dt)
