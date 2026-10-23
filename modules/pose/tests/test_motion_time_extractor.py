"""Tests for MotionTimeExtractor — accumulates AngleVelocity over time into MotionTime,
covering the non-finite velocity guard, the zero-dt tick, and the first-frame stamp."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import AngleVelocity, MotionTime
from modules.pose.nodes import MotionTimeExtractor


def _vel_frame(velocity: float, t: float) -> Frame:
    n = len(AngleVelocity.enum())
    vel = AngleVelocity(np.full(n, velocity, dtype=np.float32), np.full(n, 1.0, dtype=np.float32))
    return Frame(track_id=0, cam_id=0, time_stamp=t, features={AngleVelocity: vel})


class MotionTimeRobustnessTest(unittest.TestCase):
    def test_inf_velocity_zero_dt_never_poisons(self) -> None:
        ex = MotionTimeExtractor()
        ex.process(_vel_frame(0.0, t=1.0))                     # establishes prev_time_stamp
        out = ex.process(_vel_frame(np.inf, t=1.0))            # inf velocity on a dt==0 tick
        self.assertFalse(math.isnan(out[MotionTime].value))
        later = ex.process(_vel_frame(1.0, t=2.0))             # normal tick still accumulates finitely
        self.assertTrue(math.isfinite(later[MotionTime].value))
        self.assertGreater(later[MotionTime].value, 0.0)

    def test_inf_velocity_positive_dt_ignored(self) -> None:
        ex = MotionTimeExtractor()
        ex.process(_vel_frame(0.0, t=1.0))
        out = ex.process(_vel_frame(np.inf, t=2.0))            # dt>0 but inf is masked out
        self.assertEqual(out[MotionTime].value, 0.0)

    def test_motion_time_stamped_from_first_frame(self) -> None:
        ex = MotionTimeExtractor()
        out = ex.process(_vel_frame(1.0, t=1.0))
        self.assertIn(MotionTime, out)
        self.assertEqual(out[MotionTime].value, 0.0)


if __name__ == "__main__":
    unittest.main()
