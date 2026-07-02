"""Tests for Ghoster — the active/passive model. Each live person has a passive ghost that follows him
sweep-by-sweep, building the ghost's own dwell (sweeps on spot) and motion (MotionTime). When he moves
> band_degrees in one sweep (breaks free) *and* the passive had settled (dwell & motion full), it is
committed as an ACTIVE ghost frozen at the spot; otherwise nothing. Ghosts are frames carrying a
GhostState feature; only ACTIVE ones reach OSC. Active ghosts fade / are reclaimed on sweeps. The Ghoster
also stamps dwell/motion/fade (as GhostFeature) onto live frames."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame, reidentify
from modules.pose.features import Angles, Azimuth, MotionTime
from apps.white_space.pose import (
    GhostElement, GhostFeature, Ghoster, GhosterSettings, GhostState, GhostStateValue, PlayheadOffset,
    ghost_state,
)

_NUM_JOINTS = len(Angles.enum())


def _angles(value: float) -> Angles:
    return Angles(np.full(_NUM_JOINTS, value, dtype=np.float32),
                  np.full(_NUM_JOINTS, 1.0, dtype=np.float32))


def _live(track_id: int, azimuth: float, offset: float = -0.1, angle: float = 0.5,
          motion_time: float = 0.0) -> Frame:
    return Frame(track_id=track_id, cam_id=0, features={
        Azimuth: Azimuth.from_value(azimuth),
        PlayheadOffset: PlayheadOffset.from_value(offset),
        MotionTime: MotionTime.from_value(motion_time),
        Angles: _angles(angle),
    })


def _sweep(ghoster: Ghoster, poses: dict[int, float | tuple[float, float]], motion_time: float = 0.0) -> None:
    """One playhead sweep across each person in ``poses`` (tid -> az, or tid -> (az, angle)); the sweep
    sample fires on the −0.1 frame. Each frame carries ``motion_time``."""
    def build(offset: float) -> dict[int, Frame]:
        frames: dict[int, Frame] = {}
        for tid, p in poses.items():
            az, angle = p if isinstance(p, tuple) else (p, 0.5)
            frames[tid] = _live(tid, az, offset, angle, motion_time)
        return frames
    ghoster.process(build(0.1))     # approaching
    ghoster.process(build(-0.1))    # crossed zero → sweep sample fires here
    ghoster.process(build(3.0))     # far side, so next sweep's +0.1 isn't a false crossing


def _settle(ghoster: Ghoster, tid: int, spot: float, angle: float = 0.5) -> None:
    """Four sweeps on ``spot`` with motion accumulating → the passive ghost settles (dwell & motion full)
    under the default settings (dwell_sweeps 4, motion_sweeps 2, motion_scale 5.0)."""
    _sweep(ghoster, {tid: (spot, angle)}, motion_time=0.0)   # sweep 1
    _sweep(ghoster, {tid: (spot, angle)}, motion_time=0.0)   # sweep 2 → motion anchor = 0
    _sweep(ghoster, {tid: (spot, angle)}, motion_time=6.0)   # sweep 3 → motion 1.2
    _sweep(ghoster, {tid: (spot, angle)}, motion_time=6.0)   # sweep 4 → dwell full → settled


def _make_active(ghoster: Ghoster, tid: int = 0, spot: float = 1.0, leave: float = 2.0,
                 angle: float = 0.5) -> None:
    """Settle at ``spot`` then break free to ``leave`` → one ACTIVE ghost committed at ``spot``."""
    _settle(ghoster, tid, spot, angle)
    _sweep(ghoster, {tid: (leave, angle)}, motion_time=6.0)


def _ghost_sweep(ghoster: Ghoster, az: float, ph: dict[str, float],
                 persons: dict[int, float] | None = None) -> None:
    """Advance the ``ph`` playhead across an active ghost fixed at ``az`` — one ghost crossing. With no
    ``persons`` the ghost fades a step; a ``persons`` (tid -> az) member within band reclaims it."""
    for offset in (0.1, -0.1, 3.0):
        ph["v"] = az - offset
        frames = {tid: _live(tid, paz, offset=5.0) for tid, paz in (persons or {}).items()}
        ghoster.process(frames)


def _active(ghosts: dict[int, Frame]) -> dict[int, Frame]:
    return {gid: g for gid, g in ghosts.items() if ghost_state(g) is GhostStateValue.ACTIVE}


def _passive(ghosts: dict[int, Frame]) -> dict[int, Frame]:
    return {gid: g for gid, g in ghosts.items() if ghost_state(g) is GhostStateValue.PASSIVE}


class _Capture:
    """Captures Ghoster's three output channels (last dict emitted on each)."""

    def __init__(self, ghoster: Ghoster) -> None:
        self.tagged: dict[int, Frame] = {}
        self.ghosts: dict[int, Frame] = {}
        self.sound: dict[int, Frame] = {}
        ghoster.add_frames_callback(lambda d: setattr(self, "tagged", d))
        ghoster.add_ghosts_callback(lambda d: setattr(self, "ghosts", d))
        ghoster.add_sound_callback(lambda d: setattr(self, "sound", d))


class ReidentifyTest(unittest.TestCase):
    def test_clones_features_under_new_id(self) -> None:
        src = Frame(track_id=2, cam_id=1, features={Azimuth: Azimuth.from_value(0.5), Angles: _angles(0.3)})
        ghost = reidentify(src, 9)
        self.assertEqual(ghost.track_id, 9)
        self.assertEqual(ghost.cam_id, 1)
        self.assertAlmostEqual(ghost[Azimuth].value, 0.5, places=5)
        self.assertIn(Angles, ghost)
        self.assertEqual(src.track_id, 2)   # original untouched


class GhosterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ghoster = Ghoster(GhosterSettings(), playhead=lambda: 0.0)   # band 10°, constant playhead
        self.cap = _Capture(self.ghoster)

    def tearDown(self) -> None:
        self.ghoster.stop()

    def test_settled_break_free_leaves_active(self) -> None:
        _make_active(self.ghoster, 0, spot=1.0, leave=2.0)
        active = _active(self.cap.ghosts)
        self.assertEqual(len(active), 1)
        self.assertAlmostEqual(next(iter(active.values()))[Azimuth].value, 1.0, places=4)

    def test_break_free_commits_without_a_sweep(self) -> None:
        # Once settled, a single non-crossing tick with the person moved > band commits immediately —
        # no waiting for the playhead to reach him again.
        _settle(self.ghoster, 0, 1.0)
        self.ghoster.process({0: _live(0, 2.0, offset=5.0)})   # offset 5.0 → no crossing this tick
        active = _active(self.cap.ghosts)
        self.assertEqual(len(active), 1)
        self.assertAlmostEqual(next(iter(active.values()))[Azimuth].value, 1.0, places=4)

    def test_slow_drift_leaves_nothing(self) -> None:
        # Steps < band each sweep (motion growing) → the passive follows, never breaks free → no active.
        for i in range(6):
            _sweep(self.ghoster, {0: i * 0.1}, motion_time=float(i) * 3.0)
        self.assertEqual(len(_active(self.cap.ghosts)), 0)
        self.assertEqual(len(_passive(self.cap.ghosts)), 1)   # a passive ghost follows him

    def test_break_free_before_dwell_leaves_nothing(self) -> None:
        _sweep(self.ghoster, {0: 1.0}, motion_time=6.0)   # only 2 sweeps on the spot (dwell not full)
        _sweep(self.ghoster, {0: 1.0}, motion_time=6.0)
        _sweep(self.ghoster, {0: 2.0}, motion_time=6.0)   # break free
        self.assertEqual(len(_active(self.cap.ghosts)), 0)

    def test_break_free_without_motion_leaves_nothing(self) -> None:
        _settle_no_motion = lambda az: _sweep(self.ghoster, {0: az}, motion_time=0.0)
        for _ in range(4):
            _settle_no_motion(1.0)          # dwell fills but MotionTime never grows → motion 0
        _settle_no_motion(2.0)              # break free
        self.assertEqual(len(_active(self.cap.ghosts)), 0)

    def test_passive_present_and_not_in_sound(self) -> None:
        _sweep(self.ghoster, {0: 1.0})                    # one sweep → a passive ghost exists
        passive = _passive(self.cap.ghosts)
        self.assertEqual(len(passive), 1)
        self.assertNotIn(0, {gid for gid in self.cap.sound if ghost_state(self.cap.sound[gid]) is GhostStateValue.PASSIVE})
        # passive is keyed by the live id and must NOT be an extra voice in the sound dict
        self.assertIsNone(ghost_state(self.cap.sound[0]))   # id 0 in sound is the live pose, not a passive ghost

    def test_active_reaches_sound_passive_does_not(self) -> None:
        _make_active(self.ghoster, 0, spot=1.0, leave=2.0)
        active_ids = set(_active(self.cap.ghosts))
        self.assertTrue(active_ids)
        self.assertTrue(active_ids <= set(self.cap.sound))          # active ghosts are in OSC
        # no PASSIVE-marked frame ever reaches the sound dict (only live poses + active ghosts do)
        self.assertFalse(any(ghost_state(f) is GhostStateValue.PASSIVE for f in self.cap.sound.values()))

    def test_live_frames_carry_dwell_motion(self) -> None:
        _settle(self.ghoster, 0, 1.0)                     # settle without breaking free
        g = self.cap.tagged[0][GhostFeature]
        self.assertGreaterEqual(g.get(GhostElement.Dwell, 0.0), 1.0)
        self.assertGreaterEqual(g.get(GhostElement.Motion, 0.0), 1.0)

    def test_clear_empties_registry(self) -> None:
        _make_active(self.ghoster, 0, spot=1.0, leave=2.0)
        self.assertEqual(len(_active(self.cap.ghosts)), 1)
        self.ghoster.clear()
        self.ghoster.process({})
        self.assertEqual(len(self.cap.ghosts), 0)

    def test_num_ghosts_counts_active(self) -> None:
        _make_active(self.ghoster, 0, spot=1.0, leave=2.0)
        self.assertEqual(self.ghoster._settings.num_ghosts, 1)
        self.ghoster.clear()
        self.assertEqual(self.ghoster._settings.num_ghosts, 0)

    def test_disabled_bypasses(self) -> None:
        self.ghoster._settings.enabled = False
        _make_active(self.ghoster, 0, spot=1.0, leave=2.0)
        self.assertEqual(len(self.cap.ghosts), 0)         # nothing recorded while disabled
        self.assertIn(0, self.cap.sound)                  # live passed through
        self.assertIsNone(ghost_state(self.cap.sound[0]))


class ActiveGhostTest(unittest.TestCase):
    """Fade / reclaim / offset-refresh of active ghosts (moving playhead)."""

    def _ghoster_with_active_at_1(self, **settings_kw) -> tuple[Ghoster, _Capture, dict[str, float]]:
        settings = GhosterSettings()
        for k, v in settings_kw.items():
            setattr(settings, k, v)
        ph = {"v": 5.0}                                    # playhead parked away from the ghost
        ghoster = Ghoster(settings, playhead=lambda: ph["v"])
        cap = _Capture(ghoster)
        _make_active(ghoster, 0, spot=1.0, leave=2.0)      # active ghost at 1.0
        return ghoster, cap, ph

    def test_offset_tracks_playhead(self) -> None:
        ghoster, cap, ph = self._ghoster_with_active_at_1()
        gid = next(iter(_active(cap.ghosts)))
        ph["v"] = 0.0
        ghoster.process({})
        self.assertAlmostEqual(cap.ghosts[gid][PlayheadOffset].value, 1.0, places=4)   # az − playhead
        ph["v"] = 0.5
        ghoster.process({})
        self.assertAlmostEqual(cap.ghosts[gid][PlayheadOffset].value, 0.5, places=4)
        ghoster.stop()

    def test_reclaim_only_at_sweep(self) -> None:
        ghoster, cap, ph = self._ghoster_with_active_at_1()
        self.assertEqual(len(_active(cap.ghosts)), 1)
        for _ in range(3):                                 # person on the ghost, no sweep across it
            ghoster.process({3: _live(3, 1.0, offset=5.0)})
        self.assertEqual(len(_active(cap.ghosts)), 1)      # not deleted every frame
        _ghost_sweep(ghoster, 1.0, ph, persons={3: 1.0})   # swept while overlapped → reclaimed
        self.assertEqual(len(_active(cap.ghosts)), 0)
        ghoster.stop()

    def test_just_made_not_deleted(self) -> None:
        # A person standing where the ghost is committed does not delete it on its birth tick.
        ghoster, cap, ph = self._ghoster_with_active_at_1()
        ph["v"] = 1.0                                       # playhead sits on the ghost
        ghoster.process({3: _live(3, 1.0, offset=5.0)})     # person 3 overlaps; ghost has no prev-offset yet
        self.assertEqual(len(_active(cap.ghosts)), 1)
        ghoster.stop()

    def test_fades_out_over_fade_sweeps(self) -> None:
        ghoster, cap, ph = self._ghoster_with_active_at_1(fade_sweeps=4)
        gid = next(iter(_active(cap.ghosts)))
        ph["v"] = 0.0
        ghoster.process({})
        self.assertAlmostEqual(cap.ghosts[gid][GhostFeature].get(GhostElement.Fade), 1.0, places=5)
        for _ in range(4):                                  # 4 sweeps with no one overlapping → fade to 0
            _ghost_sweep(ghoster, 1.0, ph)
        self.assertEqual(len(_active(cap.ghosts)), 0)
        ghoster.stop()


if __name__ == "__main__":
    unittest.main()
