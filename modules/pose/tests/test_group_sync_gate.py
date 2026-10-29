"""The group-sync gate: the band's hysteresis and the mode, on the record."""

import math
import unittest

from modules.pose.analytics import GroupSimilarityMode
from modules.pose.features import GroupSimilarity, GroupSync, Similarity
from modules.pose.frame import FrameDict, GroupRecord
from modules.pose.nodes import GroupSyncGate, GroupSyncSettings

from ._builders import frame, scalar


def _pair(sim: float, third: float | None = None) -> FrameDict:
    """Two poses reading ``sim`` toward each other; with ``third`` a third pose reading that toward both."""
    frames = {0: frame(0, features={Similarity: scalar(Similarity, {1: sim, **({2: third} if third is not None else {})})}),
              1: frame(1, features={Similarity: scalar(Similarity, {0: sim, **({2: third} if third is not None else {})})})}
    if third is not None:
        frames[2] = frame(2, features={Similarity: scalar(Similarity, {0: third, 1: third})})
    return frames


class GroupSyncGateTest(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = GroupSyncSettings()
        self.settings.on_at, self.settings.off_at = 0.95, 0.9
        self.gate = GroupSyncGate(self.settings)

    def _run(self, frames: FrameDict) -> GroupRecord:
        return self.gate.record(frames, GroupRecord())

    def test_the_band_has_hysteresis_and_a_reset_forgets_the_hold(self) -> None:
        self.assertFalse(self._run(_pair(0.92))[GroupSync].on)       # inside the band from below: off
        self.assertTrue(self._run(_pair(1.0))[GroupSync].on)         # at on_at: on
        self.assertTrue(self._run(_pair(0.92))[GroupSync].on)        # hovering: held
        self.assertFalse(self._run(_pair(0.85))[GroupSync].on)       # below the band: off
        self.assertFalse(self._run(_pair(0.92))[GroupSync].on)       # back inside: stays off
        self._run(_pair(1.0))
        self.gate.reset()
        self.assertFalse(self._run(_pair(0.92))[GroupSync].on)

    def test_a_level_that_cannot_be_read_closes_the_gate(self) -> None:
        self._run(_pair(1.0))
        group = self._run({0: frame(0)})                              # alone: no pair
        self.assertTrue(math.isnan(group[GroupSimilarity].value))
        self.assertFalse(group[GroupSync].on)
        self._run(_pair(1.0))
        self.assertFalse(self._run(_pair(1.0, third=math.nan))[GroupSync].on)   # a pair with no reading

    def test_all_but_one_lets_one_unlike_pose_pass(self) -> None:
        self.assertFalse(self._run(_pair(1.0, third=0.5))[GroupSync].on)
        self.settings.mode = GroupSimilarityMode.ALL_BUT_ONE
        group = self._run(_pair(1.0, third=0.5))
        self.assertEqual(group[GroupSimilarity].value, 1.0)
        self.assertTrue(group[GroupSync].on)


if __name__ == "__main__":
    unittest.main()
