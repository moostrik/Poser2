"""blue_noise: every threshold once, and the low frequencies nearly empty."""

import unittest

import numpy as np

from modules.gl.blue_noise import blue_noise


class TestBlueNoise(unittest.TestCase):

    def setUp(self) -> None:
        self.size: int = 32
        self.map: np.ndarray = blue_noise(self.size)

    def test_every_rank_once(self) -> None:
        ranks = np.sort((self.map * self.size * self.size).round().astype(int).ravel())
        np.testing.assert_array_equal(ranks, np.arange(self.size * self.size))

    def test_the_low_frequencies_hold_little_power(self) -> None:
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(self.map - self.map.mean()))) ** 2
        y, x = np.indices(spectrum.shape) - self.size // 2
        radius = np.hypot(x, y)
        low = spectrum[(radius > 0) & (radius < self.size / 8)].mean()
        high = spectrum[radius > self.size / 4].mean()
        self.assertLess(low, high * 0.1)

    def test_the_same_seed_gives_the_same_map(self) -> None:
        np.testing.assert_array_equal(self.map, blue_noise(self.size))


if __name__ == '__main__':
    unittest.main()
