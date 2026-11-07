"""The turbulence's noise tile: divergence-free, in its band, rms 1, periodic."""
import unittest

import numpy as np

from modules.flow import swirl_noise


def divergence(field: np.ndarray) -> np.ndarray:
    """The spectral divergence of a periodic (size, size, size, 3) field, (x, y, z) components."""
    size = field.shape[0]
    k = np.fft.fftfreq(size)
    kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
    i2pi = 2j * np.pi
    spectra = [np.fft.fftn(field[..., c]) for c in range(3)]
    return np.real(np.fft.ifftn(i2pi * (kx * spectra[0] + ky * spectra[1] + kz * spectra[2])))


class SwirlNoiseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tile = swirl_noise(32, 4.0, 8.0, seed=3)

    def test_it_neither_gathers_nor_spreads(self) -> None:
        self.assertLess(float(np.abs(divergence(self.tile)).max()), 1e-6)

    def test_its_rms_is_one(self) -> None:
        self.assertAlmostEqual(float(np.sqrt(np.mean(self.tile ** 2))), 1.0, places=5)

    def test_it_holds_only_its_band(self) -> None:
        k = np.fft.fftfreq(32)
        kz, ky, kx = np.meshgrid(k, k, k, indexing='ij')
        magnitude = np.sqrt(kx * kx + ky * ky + kz * kz)
        power = np.abs(np.fft.fftn(self.tile[..., 0])) ** 2
        outside = power[(magnitude < 1.0 / 8.0 - 1e-9) | (magnitude > 1.0 / 4.0 + 1e-9)].sum()
        self.assertLess(float(outside / power.sum()), 1e-9)

    def test_the_same_seed_is_the_same_tile(self) -> None:
        np.testing.assert_array_equal(self.tile, swirl_noise(32, 4.0, 8.0, seed=3))


if __name__ == '__main__':
    unittest.main()
