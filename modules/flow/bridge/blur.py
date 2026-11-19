"""The lowpass blur's values without GL: one step's sigma, its dispatches and the Gaussian's taps."""

import math

import numpy as np

FULL_WIDTH_HALF_MAXIMUM: float = 2.0 * math.sqrt(2.0 * math.log(2.0))     # a Gaussian's width at half its height, in sigmas


def blur_sigma(blur: float, delta_time: float, width: int) -> float:
    """One step's sigma in pixels: a spread of `blur` widths at half its height in one second, as diffusion
    (σ² = 2 D dt), over a step of `delta_time` s, on a field `width` pixels wide; 0 without a spread or a step."""
    if blur <= 0.0 or delta_time <= 0.0 or not math.isfinite(delta_time):
        return 0.0
    return blur / FULL_WIDTH_HALF_MAXIMUM * math.sqrt(delta_time) * width


def blur_dispatches(sigma: float, max_radius: int) -> tuple[int, float]:
    """How many blurs of which sigma make one of `sigma` with each reaching 3 sigmas within `max_radius` taps:
    k blurs of sigma / sqrt(k) are one of sigma."""
    count: int = max(1, math.ceil((3.0 * sigma / max_radius) ** 2))
    return count, sigma / math.sqrt(count)


def gaussian_weights(sigma: float) -> np.ndarray:
    """The textbook Gaussian's taps k = 0..ceil(3 sigma): exp(-k² / 2σ²), normalized over both sides."""
    if sigma <= 0.0:
        return np.ones(1)
    radius: int = math.ceil(3.0 * sigma)
    k: np.ndarray = np.arange(radius + 1, dtype=np.float64)
    weights: np.ndarray = np.exp(-k * k / (2.0 * sigma * sigma))
    return weights / (weights[0] + 2.0 * weights[1:].sum())
