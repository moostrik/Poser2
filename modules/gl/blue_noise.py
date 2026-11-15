"""blue_noise: a tileable blue-noise threshold map, by Ulichney's void-and-cluster method (1993)."""

import numpy as np

SIGMA: float = 1.5      # the energy filter's Gaussian, in pixels (Ulichney's value)


def _energy_filter(size: int) -> np.ndarray:
    """The Gaussian's spectrum over a torus of `size` x `size`, so a convolution with it wraps around."""
    d: np.ndarray = np.minimum(np.arange(size), size - np.arange(size)).astype(np.float64)
    kernel: np.ndarray = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2.0 * SIGMA * SIGMA))
    return np.fft.rfft2(kernel)


def _energy(pattern: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    return np.fft.irfft2(np.fft.rfft2(pattern) * spectrum, s=pattern.shape)


def blue_noise(size: int = 64, seed: int = 0) -> np.ndarray:
    """A `size` x `size` float32 map of every rank 0..size²-1 once, divided by size²: thresholds whose ones, taken
    below any level, lie as evenly as they can (blue noise)."""
    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = size * size
    spectrum: np.ndarray = _energy_filter(size)

    # The initial pattern: a tenth of the pixels, its tightest cluster moved to its largest void until stable
    pattern: np.ndarray = np.zeros((size, size))
    pattern.flat[rng.choice(n, n // 10, replace=False)] = 1.0
    while True:
        energy: np.ndarray = _energy(pattern, spectrum)
        cluster: int = int(np.argmax(np.where(pattern > 0, energy, -np.inf)))
        pattern.flat[cluster] = 0.0
        energy = _energy(pattern, spectrum)
        void: int = int(np.argmin(np.where(pattern > 0, np.inf, energy)))
        if void == cluster:
            pattern.flat[cluster] = 1.0
            break
        pattern.flat[void] = 1.0

    ones: int = int(pattern.sum())
    ranks: np.ndarray = np.zeros(n, dtype=np.int64)

    # Phase 1: the initial pattern's ones ranked by taking its tightest cluster away, one by one
    work: np.ndarray = pattern.copy()
    for rank in range(ones - 1, -1, -1):
        energy = _energy(work, spectrum)
        cluster = int(np.argmax(np.where(work > 0, energy, -np.inf)))
        work.flat[cluster] = 0.0
        ranks[cluster] = rank

    # Phases 2 and 3: the rest ranked by filling the largest void, one by one
    work = pattern.copy()
    energy = _energy(work, spectrum)
    impulse: np.ndarray = np.fft.irfft2(spectrum, s=(size, size))      # one pixel's energy at the origin
    for rank in range(ones, n):
        void = int(np.argmin(np.where(work > 0, np.inf, energy)))
        work.flat[void] = 1.0
        ranks[void] = rank
        energy += np.roll(impulse, divmod(void, size), axis=(0, 1))
    return (ranks.reshape(size, size) / n).astype(np.float32)
