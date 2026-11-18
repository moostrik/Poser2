import unittest

import numpy as np

from modules.flow.color.white_balance import REFERENCE_KELVIN, RGB_TO_XYZ, planckian_white, white_balance_matrix


class TestWhiteBalance(unittest.TestCase):

    def test_reference_is_identity(self) -> None:
        np.testing.assert_allclose(white_balance_matrix(REFERENCE_KELVIN), np.eye(3), atol=1e-6)

    def test_planckian_white_known_points(self) -> None:
        # The locus's chromaticity at 2856 K (illuminant A's temperature) and 6500 K, Kim et al.'s spline
        for kelvin, xy in ((2856.0, (0.4476, 0.4074)), (6500.0, (0.3135, 0.3236))):
            white = planckian_white(kelvin)
            total = white.sum()
            self.assertAlmostEqual(white[0] / total, xy[0], delta=2e-3)
            self.assertAlmostEqual(white[1] / total, xy[1], delta=2e-3)

    def test_light_turned_white(self) -> None:
        # A warm light's white, in linear sRGB, comes out as the reference's white, its brightness kept
        for kelvin in (2700.0, 4000.0, 9000.0):
            light = np.linalg.solve(RGB_TO_XYZ, planckian_white(kelvin))
            reference = np.linalg.solve(RGB_TO_XYZ, planckian_white(REFERENCE_KELVIN))
            np.testing.assert_allclose(white_balance_matrix(kelvin) @ light, reference, atol=1e-4)

    def test_lower_cools(self) -> None:
        grey = np.array([0.5, 0.5, 0.5])
        cooled = white_balance_matrix(3000.0) @ grey
        warmed = white_balance_matrix(10000.0) @ grey
        self.assertGreater(cooled[2], cooled[0])
        self.assertGreater(warmed[0], warmed[2])


if __name__ == '__main__':
    unittest.main()
