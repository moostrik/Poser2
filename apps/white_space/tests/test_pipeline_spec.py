"""White Space's pipeline declaration builds, and the dummy comes out of LERP as a pose like any other."""

import tempfile
import unittest
from pathlib import Path

from modules.pose import Age, Angles, FrameDict, GroupRecord, GroupSync, LegDeviation, MotionGate, TorsoTilt, \
                         configure_features
from modules.pose.pipeline import Domain, PosePipeline

from apps.white_space.pose import Dummy, PlayheadOffset, dummy_id
from apps.white_space.pose.pipeline import pipeline_spec
from apps.white_space.settings import Settings, Stage


class _Source:
    def __init__(self, track_ids: list[int]) -> None:
        self.track_ids = track_ids

    def add_frames_callback(self, callback) -> None:
        pass


class _Async:
    def __init__(self) -> None:
        self.submitted: list[FrameDict] = []

    def submit(self, frames: FrameDict) -> None:
        self.submitted.append(frames)

    def add_frames_callback(self, callback) -> None:
        pass


class PipelineSpecTest(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings()
        ps = self.settings.pose
        self.players = self.settings.max_players
        configure_features(self.players)                     # the track features' width, as main.py sets it
        self.dummy = Dummy(self.settings.pose_instrument.dummy, ps.angle_extractor, ps.angle_calibrator,
                           dummy_id(self.players), Path(self.tmp.name) / 'poses.json')
        self.pipeline = PosePipeline(pipeline_spec(ps, _Source(list(range(self.players))), _Async(), self.dummy,
                                                   lambda _cam, x: x, lambda: 0.25))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_the_stages_are_the_apps_in_order(self) -> None:
        self.assertEqual(self.pipeline.stages, list(Stage))
        for stage in Stage:
            self.pipeline.window(stage)                       # every stage has its window for the board

    def test_lerp_holds_the_players_and_the_dummy(self) -> None:
        self.assertEqual(self.pipeline.track_ids(Stage.PREDICT), list(range(self.players)))
        self.assertEqual(self.pipeline.track_ids(Stage.LERP), list(range(self.players)) + [dummy_id(self.players)])

    def test_the_dummy_comes_out_of_lerp_with_everything_a_person_has(self) -> None:
        out: list[FrameDict] = []
        groups: list[GroupRecord] = []
        self.pipeline.add_frames_callback(Stage.LERP, out.append)
        self.pipeline.add_group_callback(Stage.LERP, groups.append)
        self.settings.pose_instrument.dummy.enabled = True
        self.settings.pose_instrument.dummy.morph = 0.0
        self.pipeline.update(Domain.OUTPUT, 1.0)
        self.pipeline.update(Domain.OUTPUT, 1.5)             # the age counts from the second tick
        self.assertEqual(set(out[-1]), {dummy_id(self.players)})
        f = out[-1][dummy_id(self.players)]
        for ft in (Angles, LegDeviation, TorsoTilt, Age, MotionGate, PlayheadOffset):
            self.assertIn(ft, f)
        self.assertIn(GroupSync, groups[-1])                 # the gate is on the LERP record (closed: the dummy alone)


if __name__ == "__main__":
    unittest.main()
