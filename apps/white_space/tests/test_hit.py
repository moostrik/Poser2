"""The hit: the one crossing detector stamps ``Hit`` on the pose the playhead is closest to, once per pass."""

import math
import unittest

from modules.pose import Frame, FrameDict

from apps.white_space.light import LightSettings
from apps.white_space.pose import Hit, HitExtractor, PlayheadOffset


class HitExtractorTest(unittest.TestCase):
    """At 36 rpm and 30 Hz the playhead steps 7.2° per tick."""

    def setUp(self) -> None:
        self.light = LightSettings()
        self.light.motor.beam_rpm = 36.0
        self.light.light_rate = 30.0
        self.extractor = HitExtractor(self.light)

    def _sweep(self, *offsets_deg: float) -> list[bool]:
        """Run one pose through a sequence of offsets; the hit per tick."""
        hits = []
        for deg in offsets_deg:
            frames: FrameDict = {0: Frame(0, 0, timestamp=0.0, features={PlayheadOffset: PlayheadOffset.from_value(math.radians(deg))})}
            hits.append(self.extractor.process(frames)[0][Hit].on)
        return hits

    STEADY = (23.4, 16.2, 9.0, 1.8, -5.4, -12.6)     # a steady sweep, closest tick at +1.8°

    def test_the_closest_tick_is_the_hit_once_per_pass(self) -> None:
        self.assertEqual(self._sweep(*self.STEADY), [False, False, False, True, False, False])
        self.assertEqual(self._sweep(-100.0, 100.0, 9.0, 1.8), [False, False, False, True])   # the next pass again

    def test_a_jittered_pass_hits_exactly_once(self) -> None:
        # +3.0° then −2.9° (a tracking correction): both within half a step, one hit. +4.0° and −4.0° are each
        # more than half a 7.2° step out: the sign flip fires.
        self.assertEqual(self._sweep(10.2, 3.0, -2.9, -10.1), [False, True, False, False])
        self.extractor.reset()
        self.assertEqual(self._sweep(12.0, 4.0, -4.0, -12.0), [False, False, True, False])

    def test_no_history_needed_after_a_reset(self) -> None:
        # INTRO resets on the very tick of the hit; that tick is still the hit.
        self._sweep(1.8)
        self.extractor.reset()
        self.assertEqual(self._sweep(0.5), [True])

    def test_a_pose_without_an_offset_reads_off(self) -> None:
        frames: FrameDict = {0: Frame(0, 0, timestamp=0.0)}
        self.assertFalse(self.extractor.process(frames)[0][Hit].on)


if __name__ == "__main__":
    unittest.main()
