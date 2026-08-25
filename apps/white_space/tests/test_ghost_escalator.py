"""Tests for GhostEscalator — counts ghosts ever made and latches the motor LOW→HIGH at the threshold.

Uses plain settings objects (no hardware): a ``GhostEscalatorSettings`` and a ``MotorSettings``. Each
``on_new_ghost`` call stands in for one ghost the Ghoster committed; the frame itself is ignored."""

import unittest

from modules.pose.frame import Frame
from apps.white_space.escalator import GhostEscalator, GhostEscalatorSettings
from apps.white_space.light.motor import MotorSettings, MotorMode


def _ghost() -> Frame:
    return Frame(track_id=0, cam_id=0, features={})


class GhostEscalatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = GhostEscalatorSettings()
        self.motor = MotorSettings()
        self.motor.mode = MotorMode.LOW
        self.esc = GhostEscalator(self.settings, self.motor)

    def tearDown(self) -> None:
        self.esc.stop()

    def _make(self, n: int) -> None:
        for _ in range(n):
            self.esc.on_new_ghost(_ghost())

    def test_below_threshold_stays_low(self) -> None:
        self.settings.high_threshold = 5
        self._make(4)
        self.assertEqual(self.motor.mode, MotorMode.LOW)
        self.assertEqual(self.settings.total_ghosts, 4)

    def test_reaching_threshold_escalates_to_high(self) -> None:
        self.settings.high_threshold = 5
        self._make(5)
        self.assertEqual(self.motor.mode, MotorMode.HIGH)
        self.assertEqual(self.settings.total_ghosts, 5)

    def test_latch_does_not_reassert(self) -> None:
        # After escalating, an operator sets it back to LOW; further ghosts do NOT re-force HIGH.
        self.settings.high_threshold = 2
        self._make(2)
        self.assertEqual(self.motor.mode, MotorMode.HIGH)
        self.motor.mode = MotorMode.LOW          # operator override
        self._make(5)
        self.assertEqual(self.motor.mode, MotorMode.LOW)

    def test_disabled_counts_but_never_escalates(self) -> None:
        self.settings.high_threshold = 2
        self.settings.enabled = False
        self._make(10)
        self.assertEqual(self.motor.mode, MotorMode.LOW)
        self.assertEqual(self.settings.total_ghosts, 10)   # still tallies

    def test_non_low_mode_not_stomped(self) -> None:
        # If the motor isn't in LOW when the threshold is reached, it isn't forced to HIGH.
        self.settings.high_threshold = 2
        self.motor.mode = MotorMode.STOPPED
        self._make(5)
        self.assertEqual(self.motor.mode, MotorMode.STOPPED)

    def test_reset_clears_demotes_and_rearms(self) -> None:
        self.settings.high_threshold = 2
        self._make(2)
        self.assertEqual(self.motor.mode, MotorMode.HIGH)
        GhostEscalatorSettings.reset.fire(self.settings)   # press the reset button
        self.assertEqual(self.settings.total_ghosts, 0)
        self.assertEqual(self.motor.mode, MotorMode.LOW)
        self._make(2)                                      # re-armed → escalates again
        self.assertEqual(self.motor.mode, MotorMode.HIGH)


if __name__ == "__main__":
    unittest.main()
