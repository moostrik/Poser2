"""HD Trio's pipeline declaration builds: the stages, their ids, the box steps before the pose model."""

import unittest

from modules.pose import nodes, BBox, Frame, FrameDict, configure_features
from modules.pose.pipeline import PosePipeline
from modules.utils import Rect

from apps.hd_trio.pose.pipeline import pipeline_spec
from apps.hd_trio.settings import Settings, Stage


class _Source:
    def __init__(self, track_ids: list[int]) -> None:
        self.track_ids = track_ids
        self._callbacks: list = []

    def add_frames_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, frames: FrameDict) -> None:
        for callback in self._callbacks:
            callback(frames)


class _Async:
    def __init__(self) -> None:
        self.submitted: list[FrameDict] = []

    def submit(self, frames: FrameDict) -> None:
        self.submitted.append(frames)

    def add_frames_callback(self, callback) -> None:
        pass


class PipelineSpecTest(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = Settings()
        self.players = self.settings.max_players
        configure_features(self.players)                     # the track features' width, as main.py sets it
        self.source = _Source(list(range(self.players)))
        self.pose_inference = _Async()
        self.pipeline = PosePipeline(pipeline_spec(self.settings.pose, self.source, self.pose_inference,
                                                   nodes.SimilarityResultHold()))

    def test_the_stages_are_the_apps_in_order(self) -> None:
        self.assertEqual(self.pipeline.stages, list(Stage))
        for stage in Stage:
            self.pipeline.window(stage)
            self.assertEqual(self.pipeline.track_ids(stage), list(range(self.players)))

    def test_the_boxes_are_filtered_before_the_pose_model(self) -> None:
        self.source.emit({0: Frame(0, 0, 0.0, features={BBox: BBox.from_rect(Rect(0.1, 0.1, 0.3, 0.8))})})
        self.assertEqual(len(self.pose_inference.submitted), 1)
        self.assertIn(BBox, self.pose_inference.submitted[0][0])


if __name__ == "__main__":
    unittest.main()
