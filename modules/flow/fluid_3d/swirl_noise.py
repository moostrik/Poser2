"""The turbulence's noise: a periodic 3D tile of divergence-free swirls in one band of scales, the curl of a
band-passed random vector field (as wavelet turbulence takes its noise band-limited, Kim et al. 2008), its rms 1."""

import numpy as np


def swirl_noise(size: int, low: float, high: float, seed: int = 0) -> np.ndarray:
    """A (size, size, size, 3) float32 tile, periodic: the curl of random vectors band-passed to wavelengths
    between `low` and `high` texels, scaled to an rms of 1 per component. Divergence-free in the tile's own
    spectral sense, so the swirls it adds neither gather nor spread the density."""
    rng = np.random.default_rng(seed)
    potential = rng.standard_normal((3, size, size, size))
    k = np.fft.fftfreq(size)                                     # cycles per texel
    kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
    magnitude = np.sqrt(kx * kx + ky * ky + kz * kz)
    band = (magnitude >= 1.0 / high) & (magnitude <= 1.0 / low)
    spectrum = np.fft.fftn(potential, axes=(1, 2, 3)) * band
    i2pi = 2j * np.pi
    ax, ay, az = spectrum[0], spectrum[1], spectrum[2]          # the vector potential's x, y, z (arrays are z, y, x)
    curl = np.stack((i2pi * (ky * az - kz * ay),
                     i2pi * (kz * ax - kx * az),
                     i2pi * (kx * ay - ky * ax)))
    field = np.real(np.fft.ifftn(curl, axes=(1, 2, 3)))
    field /= np.sqrt(np.mean(field * field)) + 1e-12
    return np.moveaxis(field, 0, -1).astype(np.float32)

