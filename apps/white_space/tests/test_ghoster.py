"""Tests for Ghoster — stability edge-trigger + hysteresis, band ownership / lock-out,
ghosted tagging and OSC muting, the three output channels, pool recycle/ignore, and the
``reidentify`` snapshot helper."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame, reidentify
from modules.pose.features import Angles, Azimuth
from apps.white_space.pose import Ghoster, GhosterSettings, GhostedFeature, PlayheadStability

_NUM_JOINTS = len(Angles.enum())
_HALF_BAND = math.radians(10.0 / 2.0)   # default band_degrees = 10° → ±5° ≈ 0.0873 rad


def _angles(value: float) -> Angles:
    return Angles(np.full(_NUM_JOINTS, value, dtype=np.float32),
                  np.full(_NUM_JOINTS, 1.0, dtype=np.float32))


def _live(track_id: int, stability: float, azimuth: float, angle: float = 0.5) -> Frame:
    return Frame(track_id=track_id, cam_id=0, features={
        Azimuth: Azimuth.from_value(azimuth),
        PlayheadStability: PlayheadStability.from_value(stability, 1.0),
        Angles: _angles(angle),
    })


class _Capture:
    """Captures Ghoster's three output channels (last dict emitted on each)."""

    def __init__(self, ghoster: Ghoster) -> None:
        self.tagged: dict[int, Frame] = {}
        self.ghosts: dict[int, Frame] = {}
        self.sound: dict[int, Frame] = {}
        ghoster.add_frames_callback(lambda d: setattr(self, "tagged", d))
        ghoster.add_ghosts_callback(lambda d: setattr(self, "ghosts", d))
        ghoster.add_sound_callback(lambda d: setattr(self, "sound", d))


def _ghost_azimuths(ghosts: dict[int, Frame]) -> set[float]:
    return {round(f[Azimuth].value, 3) for f in ghosts.values()}


class ReidentifyTest(unittest.TestCase):
    def test_clones_features_under_new_id(self) -> None:
        src = Frame(track_id=2, cam_id=1, features={Azimuth: Azimuth.from_value(0.5), Angles: _angles(0.3)})
        ghost = reidentify(src, 9)
        self.assertEqual(ghost.track_id, 9)
        self.assertEqual(ghost.cam_id, 1)
        self.assertEqual(ghost.time_stamp, src.time_stamp)
        self.assertAlmostEqual(ghost[Azimuth].value, 0.5, places=5)
        self.assertIn(Angles, ghost)
        self.assertEqual(src.track_id, 2)   # original untouched


class GhosterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ghoster = Ghoster(GhosterSettings())   # defaults: live_players=4, num_virtual=8
        self.cap = _Capture(self.ghoster)

    def tearDown(self) -> None:
        self.ghoster.stop()

    # -- trigger ----------------------------------------------------------

    def test_edge_fires_once_while_held(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})
        self.assertEqual(len(self.cap.ghosts), 1)                       # one lock
        self.ghoster.process({0: _live(0, 1.0, 0.0)})                  # still held → no second
        self.assertEqual(len(self.cap.ghosts), 1)

    def test_below_threshold_does_not_fire(self) -> None:
        self.ghoster.process({0: _live(0, 0.9, 0.0)})                  # < 0.95
        self.assertEqual(len(self.cap.ghosts), 0)

    def test_rearm_below_release_allows_second_ghost(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})                  # lock at 0
        self.ghoster.process({0: _live(0, 0.0, 0.0)})                  # stability < release → re-arm
        self.ghoster.process({0: _live(0, 1.0, 2.0)})                  # hold elsewhere → second lock
        self.assertEqual(len(self.cap.ghosts), 2)
        self.assertEqual(_ghost_azimuths(self.cap.ghosts), {0.0, 2.0})

    def test_no_rearm_while_above_release(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})                  # lock
        self.ghoster.process({0: _live(0, 0.8, 2.0)})                  # >release, moved to free arc
        self.ghoster.process({0: _live(0, 1.0, 2.0)})                  # still disarmed → no lock
        self.assertEqual(len(self.cap.ghosts), 1)

    # -- band ownership / ghosted ----------------------------------------

    def test_self_ghosted_mutes_and_tags(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})
        self.assertEqual(self.cap.tagged[0][GhostedFeature].value, 1.0)     # creator tagged ghosted
        self.assertNotIn(0, self.cap.sound)                            # muted from OSC
        self.assertIn(4, self.cap.sound)                              # ghost present (id 4)
        self.assertIn(4, self.cap.ghosts)

    def test_free_person_not_ghosted_and_audible(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})                  # ghost at 0
        self.ghoster.process({0: _live(0, 0.0, 0.0), 1: _live(1, 0.0, 2.0)})
        self.assertEqual(self.cap.tagged[1][GhostedFeature].value, 0.0)
        self.assertIn(1, self.cap.sound)

    def test_lockout_inside_claimed_band(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})                  # ghost at 0
        # person 1 holds stable just inside the band → no new ghost, tagged ghosted.
        self.ghoster.process({0: _live(0, 0.0, 0.0), 1: _live(1, 1.0, _HALF_BAND * 0.5)})
        self.assertEqual(len(self.cap.ghosts), 1)
        self.assertEqual(self.cap.tagged[1][GhostedFeature].value, 1.0)
        self.assertNotIn(1, self.cap.sound)

    def test_band_owner_wraps_across_pi(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 3.1)})                  # ghost near +π
        self.ghoster.process({0: _live(0, 0.0, 3.1), 1: _live(1, 1.0, -3.1)})  # near −π, wraps in
        self.assertEqual(len(self.cap.ghosts), 1)                      # person 1 locked out
        self.assertEqual(self.cap.tagged[1][GhostedFeature].value, 1.0)

    # -- lifecycle --------------------------------------------------------

    def test_clear_empties_registry(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})
        self.assertEqual(len(self.cap.ghosts), 1)
        self.ghoster.clear()
        self.ghoster.process({0: _live(0, 0.0, 0.0)})
        self.assertEqual(len(self.cap.ghosts), 0)

    def test_num_ghosts_status_field(self) -> None:
        self.ghoster.process({0: _live(0, 1.0, 0.0)})
        self.assertEqual(self.ghoster._settings.num_ghosts, 1)   # READ status reflects the registry
        self.ghoster.clear()
        self.assertEqual(self.ghoster._settings.num_ghosts, 0)

    def test_disabled_bypasses(self) -> None:
        self.ghoster._settings.enabled = False
        frames = {0: _live(0, 1.0, 0.0)}
        self.ghoster.process(frames)
        self.assertEqual(len(self.cap.ghosts), 0)
        self.assertEqual(self.cap.sound, frames)                       # passed through untouched
        self.assertNotIn(GhostedFeature, self.cap.tagged[0])               # no tagging


class GhosterPoolTest(unittest.TestCase):
    def _fill_three(self, recycle: bool) -> dict[int, Frame]:
        settings = GhosterSettings()
        settings.num_virtual = 2          # small pool to force the full/recycle path
        settings.recycle_oldest = recycle
        ghoster = Ghoster(settings)
        cap = _Capture(ghoster)
        for az in (0.0, 1.0, 2.0):                                     # three distinct holds
            ghoster.process({0: _live(0, 1.0, az)})
            ghoster.process({0: _live(0, 0.0, az)})                   # re-arm between holds
        ghoster.stop()
        return dict(cap.ghosts)

    def test_recycle_oldest_when_full(self) -> None:
        ghosts = self._fill_three(recycle=True)
        self.assertEqual(len(ghosts), 2)
        self.assertEqual(_ghost_azimuths(ghosts), {1.0, 2.0})          # oldest (0.0) recycled

    def test_ignore_new_when_full(self) -> None:
        ghosts = self._fill_three(recycle=False)
        self.assertEqual(len(ghosts), 2)
        self.assertEqual(_ghost_azimuths(ghosts), {0.0, 1.0})          # third hold ignored


if __name__ == "__main__":
    unittest.main()
