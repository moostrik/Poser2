"""`tilt_homography`: the picture of a level camera, and the distance it would have measured."""

import math
import unittest

import numpy as np

from modules.camera import homography_points, tilt_homography

SIZE: tuple[int, int] = (900, 720)
FOCAL: float = 538.0
LENS: list[list[float]] = [[FOCAL, 0.0, 450.0], [0.0, FOCAL, 360.0], [0.0, 0.0, 1.0]]


def project(point: tuple[float, float, float], tilt: float) -> tuple[np.ndarray, float]:
    """A level-frame point (x right, y down, z ahead) as a camera `tilt` degrees up sees it: its
    uv and its depth along the camera's axis."""
    cos_t, sin_t = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    x, y, z = point
    camera_y, camera_z = y * cos_t + z * sin_t, -y * sin_t + z * cos_t
    return np.array([(FOCAL * x / camera_z + 450.0) / 900.0, (FOCAL * camera_y / camera_z + 360.0) / 720.0]), camera_z


class TiltHomographyTest(unittest.TestCase):
    def test_level_is_the_identity(self) -> None:
        np.testing.assert_array_almost_equal(tilt_homography(LENS, SIZE, 0.0), np.eye(3))

    def test_the_whole_picture_stays_in_the_frame(self) -> None:
        for tilt in (-45.0, -30.0, -5.0, 5.0, 30.0):
            corners = homography_points(tilt_homography(LENS, SIZE, tilt), np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
            self.assertGreaterEqual(corners.min(), -1e-9, msg=f'tilt {tilt}')
            self.assertLessEqual(corners.max(), 1.0 + 1e-9, msg=f'tilt {tilt}')

    def test_a_vertical_line_comes_out_vertical(self) -> None:
        homography = tilt_homography(LENS, SIZE, -30.0)
        seen = np.array([project((0.7, height, 3.0), -30.0)[0] for height in (0.0, 0.8, 1.6)])
        self.assertAlmostEqual(float(np.ptp(homography_points(homography, seen)[:, 0])), 0.0, places=9)

    def test_a_vertical_wall_reads_as_one_distance(self) -> None:
        homography = tilt_homography(LENS, SIZE, -30.0)
        depths: list[float] = []
        for height in (0.0, 0.8, 1.6):
            uv, along_camera = project((0.0, height, 3.0), -30.0)
            depths.append(along_camera * float(homography[2] @ np.array([uv[0], uv[1], 1.0])))
        np.testing.assert_array_almost_equal(depths, [3.0, 3.0, 3.0])


if __name__ == '__main__':
    unittest.main()
