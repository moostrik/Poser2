"""The light's marks in the cube: a lamp's cube at its place, a sun's line from the center out past where
its light enters, both where the cube reads the volume."""

import unittest

import numpy as np

from modules.flow.port import LightPlace
from modules.flow.visualization.light_marks import MARK_SIZE, SUN_REACH, box_point, light_marks

BOX: tuple[float, float, float] = (1.0, 1.75, 0.5)
UNIT_CORNERS: np.ndarray = np.array([[x, y, z, 1.0] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])


def corners(mark: np.ndarray) -> np.ndarray:
    return (mark @ UNIT_CORNERS.T).T[:, :3]


class TestLightMarks(unittest.TestCase):

    def test_box_point_matches_the_cube(self) -> None:
        np.testing.assert_allclose(box_point((0.0, 0.0, 0.0), BOX), (-0.5, -0.875, 0.25))     # layer 0 at +z
        np.testing.assert_allclose(box_point((1.0, 1.0, 1.0), BOX), (0.5, 0.875, -0.25))

    def test_lamp_is_a_small_cube_at_its_place(self) -> None:
        marks = light_marks(LightPlace(True, (0.5, 0.25, 0.0)), BOX)
        self.assertEqual(len(marks), 1)
        points = corners(marks[0])
        np.testing.assert_allclose(points.mean(axis=0), box_point((0.5, 0.25, 0.0), BOX), atol=1e-12)
        np.testing.assert_allclose(points.max(axis=0) - points.min(axis=0), (MARK_SIZE,) * 3, atol=1e-12)

    def test_sun_line_runs_from_the_center_past_its_entry(self) -> None:
        entry = (0.5, 0.0, 0.5)             # from above
        line, cube = light_marks(LightPlace(False, entry), BOX)
        outer = box_point(entry, BOX) * SUN_REACH
        ends = corners(line)
        along = ends @ (outer / np.linalg.norm(outer))
        self.assertAlmostEqual(along.min(), 0.0, places=9)
        self.assertAlmostEqual(along.max(), float(np.linalg.norm(outer)), places=9)
        np.testing.assert_allclose(corners(cube).mean(axis=0), outer, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
