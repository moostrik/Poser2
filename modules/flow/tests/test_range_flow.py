"""Range flow's numpy reference: motion from depth alone, sideways and toward the camera in one equation."""

import unittest

import numpy as np

from modules.flow import range_flow

SIZE: int = 64
DT: float = 1.0 / 60.0


def bump(dx: float = 0.0, closer: float = 0.0) -> np.ndarray:
    """A rounded body 2.5 m away in front of a wall at 3 m, moved `dx` pixels right and `closer` metres toward the camera."""
    rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    r2 = ((columns - 32.0 - dx) ** 2 + (rows - 32.0) ** 2) / 12.0 ** 2
    body = 2.5 - closer + 0.15 * r2         # nearest at its middle, curving away
    return np.where(r2 < 1.0, body, 3.0)


def on_body(field: np.ndarray, dx: float = 0.0) -> np.ndarray:
    """The field where the moved body is, away from its outline."""
    rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    return field[((columns - 32.0 - dx) ** 2 + (rows - 32.0) ** 2) < 7.0 ** 2]


class RangeFlowTest(unittest.TestCase):
    def test_a_body_coming_closer_reads_toward_the_camera_and_no_sideways_motion(self) -> None:
        d, toward = range_flow(bump(), bump(closer=0.01), DT)
        self.assertAlmostEqual(float(np.median(on_body(toward))), 0.6, delta=0.05)     # 1 cm per frame
        self.assertLess(float(np.median(np.abs(on_body(d[..., 0])))), 0.2)

    def test_a_body_moving_sideways_reads_its_shift_and_no_motion_toward_the_camera(self) -> None:
        d, toward = range_flow(bump(), bump(dx=2.0), DT)
        self.assertAlmostEqual(float(np.median(on_body(d[..., 0], 2.0))), 2.0, delta=0.4)
        self.assertLess(abs(float(np.median(on_body(toward, 2.0)))), 0.1)

    def test_still_depth_is_no_motion(self) -> None:
        d, toward = range_flow(bump(), bump(), DT)
        self.assertLess(float(np.abs(d).max()), 1e-6)
        self.assertLess(float(np.abs(toward).max()), 1e-6)

    def test_no_depth_is_no_motion(self) -> None:
        empty = np.zeros((SIZE, SIZE))
        d, toward = range_flow(empty, empty, DT)
        self.assertEqual(float(np.abs(d).max()), 0.0)
        self.assertEqual(float(np.abs(toward).max()), 0.0)


if __name__ == '__main__':
    unittest.main()
