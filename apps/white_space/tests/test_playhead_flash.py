"""Tests for the INTRO flash layer: the beam lights strike on at a pose's hit, for ``frames`` ticks."""

import unittest
from types import SimpleNamespace

from apps.white_space.light import Tick, MotorCommand
from apps.white_space.light.frame import LightFrame, BeamLightId
from apps.white_space.light.layers.beam.flash import BeamFlash, BeamFlashSettings
from apps.white_space.pose import Hit

RES = 200


class FlashPose:
    """A pose frame carrying only its Hit: all the flash reads."""

    def __init__(self, track_id: int, hit: bool) -> None:
        self.track_id = track_id
        self._hit = hit

    def __getitem__(self, feature_type):
        assert feature_type is Hit
        return Hit.from_value(1.0 if self._hit else 0.0)


class FlashBoard(SimpleNamespace):
    """Poses only: the flash never reads tracklets. Records the flashes it is handed."""

    def get_frames(self, stage: int):
        return self.frames

    def add_flash(self, azimuth: float, white: float, blue: float) -> None:
        self.flashes.append((azimuth, white, blue))


class FlashTest(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = BeamFlashSettings()
        self.board = FlashBoard(frames={}, flashes=[])
        self.layer = BeamFlash(RES, self.settings, self.board, pose_stage=4)

    def _render(self, pose, playhead: float = 1.0) -> LightFrame:
        self.board.frames = {} if pose is None else {pose.track_id: pose}
        f = LightFrame(RES, Tick(0.0, 1.0 / 30.0), motor_command=MotorCommand(beam_rpm=36.0), playhead=playhead)
        self.layer.render(f)
        return f

    def _sweep(self, *hits: bool) -> list[float]:
        """Run one pose through a sequence of ticks, hit or not; the front white lamp per tick."""
        return [float(self._render(FlashPose(0, hit)).beam_lights[BeamLightId.FRONT_WHITE]) for hit in hits]

    def test_one_frame_is_the_hit_tick(self) -> None:
        self.assertEqual(self._sweep(False, True, False, False), [0.0, 1.0, 0.0, 0.0])

    def test_more_frames_hold_the_flash_over_the_ticks_after_the_hit(self) -> None:
        self.settings.frames = 3
        self.assertEqual(self._sweep(False, True, False, False, False), [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_flash_is_full_brightness_over_the_base(self) -> None:
        self.settings.white = 0.7
        self.settings.base_white = 0.2
        self.assertAlmostEqual(self._sweep(True)[0], 0.9, places=6)
        self.assertAlmostEqual(self._sweep(False)[0], 0.2, places=6)

    def test_nobody_with_a_pose_no_flash(self) -> None:
        self.assertEqual(float(self._render(None).beam_lights[BeamLightId.FRONT_WHITE]), 0.0)

    def test_each_lit_tick_posts_a_flash_at_the_playhead_and_no_playhead_posts_nothing(self) -> None:
        self.settings.frames = 2
        for hit in (False, True, False, False):
            self._render(FlashPose(0, hit), playhead=1.25)
        self.assertEqual(self.board.flashes, [(1.25, 1.0, 0.0), (1.25, 1.0, 0.0)])
        self._render(FlashPose(0, True), playhead=float("nan"))
        self.assertEqual(len(self.board.flashes), 2)

    def test_reset_ends_a_held_flash(self) -> None:
        self.settings.frames = 3
        self._sweep(True)
        self.layer.reset()
        self.assertEqual(self._sweep(False), [0.0])


if __name__ == "__main__":
    unittest.main()
