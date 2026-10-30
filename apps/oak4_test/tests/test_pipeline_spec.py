"""The OAK4 test app builds without a device: its settings, its preset and its pipeline. The app has no
simulator, so this is the one check it gets before the OAK4 is plugged in. Runs on a depthai 3 install."""

import unittest

import numpy as np

from modules.pose import BBox, Frame, Points2D, configure_features
from modules.pose.pipeline import PosePipeline
from modules.settings import presets
from modules.utils import Rect

from apps.oak4_test.pose.pipeline import pipeline_spec
from apps.oak4_test.settings import Settings, Stage


class _Source:
    def __init__(self, track_ids: list[int]) -> None:
        self.track_ids = track_ids
        self._callbacks: list = []

    def add_frames_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, frames) -> None:
        for callback in self._callbacks:
            callback(frames)


class PipelineSpecTest(unittest.TestCase):

    def test_the_preset_loads_and_the_camera_keypoints_reach_smooth(self) -> None:
        settings = Settings()
        presets.set_app('oak4_test')
        self.assertTrue(presets.load(settings, presets.startup_path()))
        players: int = settings.max_players
        configure_features(players)                          # the track features' width, as main.py sets it
        source = _Source(list(range(players)))
        pipeline = PosePipeline(pipeline_spec(settings.pose, source))
        self.assertEqual(pipeline.stages, list(Stage))

        smooth: list = []
        pipeline.add_frames_callback(Stage.SMOOTH, smooth.append)
        points = Points2D.from_xy_arrays(np.full(17, 0.5, np.float32), np.full(17, 0.5, np.float32), np.full(17, 0.9, np.float32))
        source.emit({0: Frame(0, 0, 0.0, features={BBox: BBox.from_rect(Rect(0.1, 0.1, 0.3, 0.8)), Points2D: points})})
        self.assertEqual(len(smooth), 1)
        self.assertIn(Points2D, smooth[0][0])


if __name__ == "__main__":
    unittest.main()
