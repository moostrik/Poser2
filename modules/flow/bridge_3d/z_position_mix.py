"""An output's z position from its readings' (numpy reference of z_position_mix.frag)."""

import numpy as np

from .trails import EPSILON


def z_position_mix(readings: list[np.ndarray], shares: list[float], trails: tuple[np.ndarray, np.ndarray, np.ndarray],
                   z_positions: tuple[np.ndarray, np.ndarray, np.ndarray],
                   mask: tuple[np.ndarray, np.ndarray, float] | None = None) -> np.ndarray:
    """An output's z position (..., 2, red the z position times its weight, green the weight): its readings' z
    positions weighted by each reading's size at its share, each times the share of its trail's speed that came
    with a z position. `readings` the six in order (trail, change, start, bow, stop, wake, ..., 3), `trails` the
    lowpass's trail and the bandpass's fast and slow trail (..., 3), `z_positions` theirs (..., 2). The trail
    reads the lowpass's, a start and a bow the fast trail's, a stop and a wake the slow trail's, the change the
    fast trail's where it is the faster, else the slow's. `mask`: the mask (..., 1), the motion's z position now
    (..., 2) and the mask's share, which weighs that z position by the mask at the share.
    Mirrors z_position_mix.frag."""
    speeds: list[np.ndarray] = [np.linalg.norm(trail, axis=-1, keepdims=True) for trail in trails]
    per_speed: list[np.ndarray] = [np.divide(z, s, out=np.zeros_like(z), where=s > EPSILON) for z, s in zip(z_positions, speeds)]
    sizes: list[np.ndarray] = [share * np.linalg.norm(reading, axis=-1, keepdims=True) for reading, share in zip(readings, shares)]
    at_trail, at_fast, at_slow = per_speed
    change_at: np.ndarray = np.where(speeds[1] > speeds[2], at_fast, at_slow)
    total: np.ndarray = sizes[0] * at_trail + sizes[1] * change_at + (sizes[2] + sizes[3]) * at_fast + (sizes[4] + sizes[5]) * at_slow
    if mask is not None and mask[2] > 0.0:
        presence, now, share = mask
        weight: np.ndarray = now[..., 1:2]
        place: np.ndarray = np.divide(now[..., 0:1], weight, out=np.zeros_like(weight), where=weight > EPSILON)
        total = total + share * presence * np.where(weight > EPSILON, np.concatenate((place, np.ones_like(place)), axis=-1), 0.0)
    return total
