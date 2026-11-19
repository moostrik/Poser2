"""The bandpass's band without GL: the step peak of its two trails, the band's width and the change's scale as an
acceleration."""

import math

from .trails import decay_weight


def step_peak(fast_smooth: float, slow_smooth: float) -> float:
    """The largest difference of a slow and a fast Ema (time constants in one unit) after a unit step: where
    the slow one still holds what the fast one let go. 1 with an instant fast trail, 0 when the slow
    trail is not slower. Continuous, as `decay_weight` is."""
    if slow_smooth <= fast_smooth:
        return 0.0
    if fast_smooth <= 0.0:
        return 1.0
    t: float = fast_smooth * slow_smooth / (slow_smooth - fast_smooth) * math.log(slow_smooth / fast_smooth)
    return math.exp(-t / slow_smooth) - math.exp(-t / fast_smooth)


def band_width(fast_smooth: float, slow_smooth: float, delta_time: float) -> float:
    """The band's width, s: what (fast − slow) adds up to over time, × dt, per unit of a change of speed. For the
    two Emas stepped at `delta_time` it is dt × (w_slow / (1 − w_slow) − w_fast / (1 − w_fast)), w each one's
    `decay_weight`; as dt shrinks it is (slow_smooth − fast_smooth), and it stays exact where a frame is longer
    than the fast trail. 0 when the slow trail is not slower."""
    if slow_smooth <= fast_smooth:
        return 0.0
    if delta_time <= 0.0:
        return (slow_smooth - fast_smooth) / 1000.0

    def memory(smooth: float) -> float:
        weight: float = decay_weight(delta_time, smooth) if smooth > 0.0 else 0.0
        return delta_time * weight / (1.0 - weight)
    return memory(slow_smooth) - memory(fast_smooth)


def change_acceleration_scale(fast_smooth: float, slow_smooth: float, delta_time: float) -> float:
    """What (fast − slow) is multiplied by to read as an acceleration, widths/s²: 1 over the band's width
    (`band_width`, s; the times in ms), so its sum × dt over a change of speed is that change; 0 when the slow
    trail is not slower."""
    width: float = band_width(fast_smooth, slow_smooth, delta_time)
    return 1.0 / width if width > 0.0 else 0.0
