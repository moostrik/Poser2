"""Tests for Ghoster — the per-sweep model: each time the playhead sweeps across a person (a
PlayheadOffset zero-crossing), if he moved > band_degrees since the previous sweep, a ghost is left
at that previous spot. Every tick, a live person deletes ghosts within band_degrees of himself, so a
slow drift (< band per sweep) leaves nothing and walking onto a ghost clears it. No muting. Plus pool
recycle/ignore, the ghost PlayheadOffset refresh, and the ``reidentify`` snapshot helper."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame, reidentify
from modules.pose.features import Angles, Azimuth
from apps.white_space.pose import Fade, FadeExtractor, Ghoster, GhosterSettings, PlayheadElement, PlayheadOffset, PlayheadStability

_NUM_JOINTS = len(Angles.enum())


def _angles(value: float) -> Angles:
    return Angles(np.full(_NUM_JOINTS, value, dtype=np.float32),
                  np.full(_NUM_JOINTS, 1.0, dtype=np.float32))


def _live(track_id: int, azimuth: float, offset: float = -0.1, angle: float = 0.5,
          dwell: float = 1.0, motion: float = 1.0) -> Frame:
    return Frame(track_id=track_id, cam_id=0, features={
        Azimuth: Azimuth.from_value(azimuth),
        PlayheadOffset: PlayheadOffset.from_value(offset),
        PlayheadStability: PlayheadStability(
            values=np.array([dwell, motion, 0.0], dtype=np.float32),
            scores=np.array([1.0, 1.0, 1.0], dtype=np.float32)),
        Angles: _angles(angle),
    })


def _sweep(ghoster: Ghoster, poses: dict[int, float | tuple[float, float]],
           dwell: float = 1.0, motion: float = 1.0) -> None:
    """Drive one playhead sweep across each person in ``poses`` (tid -> az, or tid -> (az, angle)):
    a PlayheadOffset that crosses zero once (the sample fires on the −0.1 frame), then a far value so
    the next sweep's approach doesn't re-trigger a crossing. Every pose carries ``dwell`` / ``motion``
    (both 1.0 = settled, so the commit gate passes by default)."""
    def build(offset: float) -> dict[int, Frame]:
        frames: dict[int, Frame] = {}
        for tid, p in poses.items():
            az, angle = p if isinstance(p, tuple) else (p, 0.5)
            frames[tid] = _live(tid, az, offset, angle, dwell, motion)
        return frames
    ghoster.process(build(0.1))    # approaching
    ghoster.process(build(-0.1))   # crossed zero → sweep sample fires here
    ghoster.process(build(3.0))    # far side, so next sweep's +0.1 isn't a false crossing


def _ghost_sweep(ghoster: Ghoster, az: float, ph: dict[str, float],
                 persons: dict[int, float] | None = None) -> None:
    """Advance the ``ph`` playhead callable across a ghost fixed at ``az`` — one ghost crossing. With no
    ``persons`` the ghost just fades a step; ``persons`` (tid -> az) are present but not themselves
    crossing (offset 5.0), so one standing within band of the ghost reclaims (deletes) it on the sweep."""
    for offset in (0.1, -0.1, 3.0):
        ph["v"] = az - offset
        frames = {tid: _live(tid, paz, offset=5.0) for tid, paz in (persons or {}).items()}
        ghoster.process(frames)


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
        self.ghoster = Ghoster(GhosterSettings(), playhead=lambda: 0.0)   # band_degrees = 10°
        self.cap = _Capture(self.ghoster)

    def tearDown(self) -> None:
        self.ghoster.stop()

    def test_moved_since_last_sweep_leaves_a_ghost(self) -> None:
        _sweep(self.ghoster, {0: 0.0})                 # first sweep — no previous position
        self.assertEqual(len(self.cap.ghosts), 0)
        _sweep(self.ghoster, {0: 0.5})                 # moved >band since last sweep → ghost at 0.0
        self.assertEqual(len(self.cap.ghosts), 1)
        self.assertEqual(_ghost_azimuths(self.cap.ghosts), {0.0})

    def test_not_left_until_settled(self) -> None:
        # Moved > band, but the pose left behind hadn't settled → no ghost (gate: Dwell & Motion == 1).
        _sweep(self.ghoster, {0: 0.0}, motion=0.5)     # on the spot, not performed (motion < 1)
        _sweep(self.ghoster, {0: 0.5}, motion=0.5)     # moved on → gate fails
        self.assertEqual(len(self.cap.ghosts), 0)
        _sweep(self.ghoster, {0: 0.0}, dwell=0.5)      # not dwelled long enough (dwell < 1)
        _sweep(self.ghoster, {0: 0.5}, dwell=0.5)
        self.assertEqual(len(self.cap.ghosts), 0)

    def test_slow_drift_leaves_nothing(self) -> None:
        # Each per-sweep step is < band, so nothing is ever left — even though the total far exceeds band.
        step = math.radians(GhosterSettings().band_degrees) * 0.6
        for i in range(6):
            _sweep(self.ghoster, {0: i * step})
        self.assertEqual(len(self.cap.ghosts), 0)

    def test_stationary_leaves_nothing(self) -> None:
        for _ in range(5):
            _sweep(self.ghoster, {0: 0.03})            # jitter well within band
        self.assertEqual(len(self.cap.ghosts), 0)

    def test_override_replaces_existing(self) -> None:
        _sweep(self.ghoster, {1: (0.0, 0.1)})
        _sweep(self.ghoster, {1: (0.5, 0.1)})          # person 1 leaves a ghost at 0.0 (angle 0.1)
        self.assertEqual(len(self.cap.ghosts), 1)
        gid = next(iter(self.cap.ghosts))
        # Person 0 settles on the same spot and leaves. With a constant playhead the old ghost is never
        # swept (so not reclaimed) — the leave-behind replaces it on the same voice via _band_owner.
        _sweep(self.ghoster, {0: (0.0, 0.9)})
        _sweep(self.ghoster, {0: (0.5, 0.9)})
        self.assertEqual(len(self.cap.ghosts), 1)
        self.assertEqual(set(self.cap.ghosts), {gid})
        self.assertAlmostEqual(float(self.cap.ghosts[gid][Angles].values[0]), 0.9, places=4)

    def test_reclaim_only_at_sweep(self) -> None:
        ph = {"v": 5.0}                                  # playhead parked away from the ghost
        ghoster = Ghoster(GhosterSettings(), playhead=lambda: ph["v"])
        cap = _Capture(ghoster)
        _sweep(ghoster, {0: 1.0})
        _sweep(ghoster, {0: 2.0})                        # ghost at 1.0
        self.assertEqual(len(cap.ghosts), 1)
        for _ in range(3):                               # person stands on the ghost, no sweep across it
            ghoster.process({3: _live(3, 1.0, offset=5.0)})
        self.assertEqual(len(cap.ghosts), 1)             # NOT deleted every frame
        self.assertIn(3, cap.sound)                      # live person audible
        _ghost_sweep(ghoster, 1.0, ph, persons={3: 1.0})  # swept while overlapped → reclaimed
        self.assertEqual(len(cap.ghosts), 0)
        ghoster.stop()

    def test_just_made_not_deleted(self) -> None:
        ph = {"v": 0.0}                                  # playhead sits on 0.0, where the ghost lands
        ghoster = Ghoster(GhosterSettings(), playhead=lambda: ph["v"])
        cap = _Capture(ghoster)

        def frames(az1: float, off1: float) -> dict[int, Frame]:
            # person 1 sweeps (crosses); person 3 stands on 0.0 without crossing (offset 5.0)
            return {1: _live(1, az1, off1), 3: _live(3, 0.0, offset=5.0)}

        for off1 in (0.1, -0.1, 3.0):
            ghoster.process(frames(0.0, off1))           # person 1 first sweep at 0.0
        for off1 in (0.1, -0.1, 3.0):
            ghoster.process(frames(0.5, off1))           # person 1 leaves → commits a ghost at 0.0
        self.assertEqual(len(cap.ghosts), 1)             # survived its birth despite person 3 on the spot
        ghoster.stop()

    def test_ghost_playhead_offset_tracks_playhead(self) -> None:
        ph = {"v": 0.0}
        ghoster = Ghoster(GhosterSettings(), playhead=lambda: ph["v"])
        cap = _Capture(ghoster)
        _sweep(ghoster, {0: 1.0})
        _sweep(ghoster, {0: 2.0})                       # commit a ghost at 1.0
        gid = next(iter(cap.ghosts))
        self.assertAlmostEqual(cap.ghosts[gid][PlayheadOffset].value, 1.0, places=4)   # az − playhead
        frozen_az = cap.ghosts[gid][Azimuth].value
        ph["v"] = 0.5
        _sweep(ghoster, {0: 2.0})                       # re-emit, same spot → no new commit
        self.assertAlmostEqual(cap.ghosts[gid][Azimuth].value, frozen_az, places=5)    # azimuth frozen
        self.assertAlmostEqual(cap.ghosts[gid][PlayheadOffset].value, 0.5, places=4)   # 1.0 − 0.5
        ghoster.stop()

    def test_clear_empties_registry(self) -> None:
        _sweep(self.ghoster, {0: 0.0})
        _sweep(self.ghoster, {0: 0.5})
        self.assertEqual(len(self.cap.ghosts), 1)
        self.ghoster.clear()
        _sweep(self.ghoster, {0: 1.0})                 # last-sweep state wiped → no commit
        self.assertEqual(len(self.cap.ghosts), 0)

    def test_num_ghosts_status_field(self) -> None:
        _sweep(self.ghoster, {0: 0.0})
        _sweep(self.ghoster, {0: 0.5})
        self.assertEqual(self.ghoster._settings.num_ghosts, 1)
        self.ghoster.clear()
        self.assertEqual(self.ghoster._settings.num_ghosts, 0)

    def test_disabled_bypasses(self) -> None:
        self.ghoster._settings.enabled = False
        _sweep(self.ghoster, {0: 0.0})
        _sweep(self.ghoster, {0: 0.5})
        self.assertEqual(len(self.cap.ghosts), 0)      # no sampling while disabled
        self.assertIn(0, self.cap.sound)               # live passed through
        self.assertEqual(len(self.cap.sound), 1)       # no ghosts added


class GhosterPoolTest(unittest.TestCase):
    def _walk(self, recycle: bool) -> dict[int, Frame]:
        settings = GhosterSettings()
        settings.num_virtual = 2                # pool of 2 → the 3rd commit hits the full path
        settings.recycle_oldest = recycle
        ghoster = Ghoster(settings, playhead=lambda: 0.0)
        cap = _Capture(ghoster)
        for az in (0.0, 1.0, 2.0, 3.0):         # each sweep moved >band → 3 commits
            _sweep(ghoster, {0: az})
        ghosts = dict(cap.ghosts)
        ghoster.stop()
        return ghosts

    def test_recycle_oldest_when_full(self) -> None:
        ghosts = self._walk(recycle=True)
        self.assertEqual(len(ghosts), 2)
        self.assertEqual(_ghost_azimuths(ghosts), {1.0, 2.0})   # oldest (0.0) recycled

    def test_ignore_new_when_full(self) -> None:
        ghosts = self._walk(recycle=False)
        self.assertEqual(len(ghosts), 2)
        self.assertEqual(_ghost_azimuths(ghosts), {0.0, 1.0})   # third commit (2.0) ignored


class GhostFadeTest(unittest.TestCase):
    def _ghoster_with_ghost_at_1(self, fade_sweeps: int) -> tuple[Ghoster, _Capture, dict[str, float], int]:
        settings = GhosterSettings()
        settings.fade_sweeps = fade_sweeps
        ph = {"v": 5.0}                                   # playhead far from the ghost while it's created
        ghoster = Ghoster(settings, playhead=lambda: ph["v"])
        cap = _Capture(ghoster)
        _sweep(ghoster, {0: 1.0})
        _sweep(ghoster, {0: 2.0})                         # leaves a ghost at 1.0
        return ghoster, cap, ph, next(iter(cap.ghosts))

    def test_fade_extractor_stamps_one(self) -> None:
        out = FadeExtractor().process(_live(0, 0.0))      # _live carries no Fade
        self.assertAlmostEqual(out[Fade].value, 1.0, places=5)

    def test_fresh_ghost_is_fully_present(self) -> None:
        _, cap, _, gid = self._ghoster_with_ghost_at_1(fade_sweeps=4)
        self.assertAlmostEqual(cap.ghosts[gid][Fade].value, 1.0, places=5)

    def test_fade_decrements_one_step_per_sweep(self) -> None:
        ghoster, cap, ph, gid = self._ghoster_with_ghost_at_1(fade_sweeps=4)
        _ghost_sweep(ghoster, 1.0, ph)
        self.assertAlmostEqual(cap.ghosts[gid][Fade].value, 0.75, places=5)
        _ghost_sweep(ghoster, 1.0, ph)
        self.assertAlmostEqual(cap.ghosts[gid][Fade].value, 0.5, places=5)
        ghoster.stop()

    def test_removed_after_fade_sweeps(self) -> None:
        ghoster, cap, ph, _ = self._ghoster_with_ghost_at_1(fade_sweeps=4)
        self.assertEqual(len(cap.ghosts), 1)
        for _ in range(4):                                # 4 × 0.25 → 0.0 → removed
            _ghost_sweep(ghoster, 1.0, ph)
        self.assertEqual(len(cap.ghosts), 0)
        ghoster.stop()

    def test_constant_playhead_never_fades(self) -> None:
        # Existing tests use a fixed playhead; a ghost is never swept, so it must not fade or vanish.
        ghoster = Ghoster(GhosterSettings(), playhead=lambda: 0.0)
        cap = _Capture(ghoster)
        _sweep(ghoster, {0: 1.0})
        _sweep(ghoster, {0: 2.0})
        gid = next(iter(cap.ghosts))
        for _ in range(5):
            ghoster.process({})
        self.assertAlmostEqual(cap.ghosts[gid][Fade].value, 1.0, places=5)
        ghoster.stop()


if __name__ == "__main__":
    unittest.main()
