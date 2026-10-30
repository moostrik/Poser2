"""`depth_metres` turns the camera's depth into what the layer uploads."""

import unittest

import numpy as np

from modules.render.layers.source.depth_source_layer import depth_metres


class DepthMetresTest(unittest.TestCase):
    def test_millimetres_become_metres_and_no_depth_stays_zero(self) -> None:
        metres = depth_metres(np.array([[0, 1500], [12000, 300]], np.uint16))
        self.assertEqual(metres.dtype, np.float32)
        np.testing.assert_allclose(metres, [[0.0, 1.5], [12.0, 0.3]])


if __name__ == '__main__':
    unittest.main()
