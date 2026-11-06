"""The 3D fluid's drag, its arithmetic in numpy: the step's share of the gap, the floor gate, and that
the water never passes the body's velocity, whatever the frame rate or the weight."""

import math
import unittest

import numpy as np

from modules.flow import DRAG_FLOOR, drag_share, drag_gate, drag_velocity_step


class DragShareTest(unittest.TestCase):
    def test_one_drag_time_closes_all_but_one_over_e(self) -> None:
        self.assertAlmostEqual(drag_share(0.05, 50.0), 1.0 - 1.0 / math.e)

    def test_steps_compose_whatever_the_frame_rate(self) -> None:
        remaining_60 = (1.0 - drag_share(1 / 60, 50.0)) ** 2
        self.assertAlmostEqual(remaining_60, 1.0 - drag_share(1 / 30, 50.0))

    def test_no_drag_time_sets_the_velocity_at_once(self) -> None:
        self.assertEqual(drag_share(1 / 60, 0.0), 1.0)


class DragGateTest(unittest.TestCase):
    def test_a_dead_trail_drags_nothing(self) -> None:
        self.assertEqual(float(drag_gate(np.array(0.0))), 0.0)

    def test_the_floor_is_the_gate_s_top(self) -> None:
        np.testing.assert_allclose(drag_gate(np.array([DRAG_FLOOR / 2, DRAG_FLOOR, 1.0])), [0.5, 1.0, 1.0])


class DragStepTest(unittest.TestCase):
    def test_the_water_moves_toward_the_body_by_the_share(self) -> None:
        got = drag_velocity_step(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), gain=1.0, share=0.25)
        np.testing.assert_allclose(got, [[0.25, 0.0, 0.0]])

    def test_the_water_never_passes_the_body_times_gain(self) -> None:
        rng = np.random.default_rng(3)
        velocity = rng.normal(size=(64, 3))
        boundary = rng.normal(size=(64, 3)) * 2.0
        v = velocity
        for _ in range(200):
            v = drag_velocity_step(v, boundary, gain=1.5, share=drag_share(1 / 30, 50.0), weight=1.0, slab=1.0)
        np.testing.assert_allclose(v, boundary * 1.5, atol=1e-6)
        bound = np.maximum(np.linalg.norm(velocity, axis=-1), np.linalg.norm(boundary * 1.5, axis=-1))
        self.assertTrue(np.all(np.linalg.norm(v, axis=-1) <= bound + 1e-9))

    def test_a_weight_and_a_slab_only_slow_the_pull(self) -> None:
        boundary = np.array([[0.0, 1.0, 0.0]])
        full = drag_velocity_step(np.zeros((1, 3)), boundary, 1.0, 1.0)
        half = drag_velocity_step(np.zeros((1, 3)), boundary, 1.0, 1.0, weight=0.5, slab=1.0)
        np.testing.assert_allclose(full, boundary)
        np.testing.assert_allclose(half, boundary * 0.5)

    def test_where_the_body_is_still_the_water_coasts(self) -> None:
        velocity = np.array([[0.3, 0.0, 0.0]])
        got = drag_velocity_step(velocity, np.zeros((1, 3)), 1.0, 1.0)
        np.testing.assert_allclose(got, velocity)


if __name__ == '__main__':
    unittest.main()
