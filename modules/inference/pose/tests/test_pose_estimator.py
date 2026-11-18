"""The pose estimator's queue on its own thread, with a fake model: a batch replaced while the model runs is counted,
not run; the batches it ran reach the callbacks from its thread, normalized to the model's input."""

import threading
import unittest

import numpy as np
import torch

from modules.inference import ModelType, crop, pose
from modules.inference.engine import Resolution
from modules.pose import BBox, Frame, Points2D
from modules.utils import Rect

from ..decode import NUM_KEYPOINTS, decode_simcc


class FakeModel:
    """The estimator's model steps without a GPU: its `_infer`, blocking until released."""
    model_width = 192
    model_height = 256

    def __init__(self) -> None:
        self.running = threading.Event()
        self.release = threading.Event()

    def infer(self, images: list[torch.Tensor], ready: torch.cuda.Event | None) -> tuple[np.ndarray, np.ndarray]:
        self.running.set()
        self.release.wait(2.0)
        n = len(images)
        simcc_x = np.zeros((n, NUM_KEYPOINTS, self.model_width * 2), np.float32)
        simcc_y = np.zeros((n, NUM_KEYPOINTS, self.model_height * 2), np.float32)
        simcc_x[:, :, 96] = 0.9         # x 48 of 192
        simcc_y[:, :, 256] = 0.8        # y 128 of 256
        return simcc_x, simcc_y


def crop_image() -> crop.Image:
    return crop.Image(crop=torch.zeros(1))


class PoseEstimatorTest(unittest.TestCase):
    def test_a_batch_replaced_while_the_model_runs_is_counted_not_run(self) -> None:
        settings = pose.PoseEstimatorSettings(model_type=ModelType.ONNX)
        settings.initialize()
        estimator = pose.PoseEstimator(settings)
        runner = FakeModel()
        estimator._setup = lambda: None
        estimator._infer = runner.infer
        told: list[tuple[list[int], str]] = []
        points: list[np.ndarray] = []
        done = threading.Event()

        def on_frames(frames) -> None:
            told.append((sorted(frames), threading.current_thread().name))
            points.extend(frame[Points2D].values[0] for frame in frames.values())
            if len(told) == 2:
                done.set()

        estimator.add_frames_callback(on_frames)
        estimator.start()
        try:
            for _ in range(200):
                if estimator.is_ready:
                    break
                threading.Event().wait(0.01)
            frame = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.0, 0.0, 1.0, 1.0))})
            estimator.submit({0: frame}, {0: crop_image()})           # runs
            self.assertTrue(runner.running.wait(2.0))
            estimator.submit({1: frame}, {1: crop_image()})           # waits
            estimator.submit({2: frame}, {2: crop_image()})           # replaces it
            runner.release.set()
            self.assertTrue(done.wait(2.0))
        finally:
            estimator.stop()

        self.assertEqual(settings.dropped, 1)
        self.assertEqual([ids for ids, _ in told], [[0], [2]])
        self.assertTrue(all(name == 'PoseEstimator' for _, name in told))
        self.assertTrue(np.allclose(points[0], [0.25, 0.5]))   # 48 of 192, 128 of 256


class DecodeTest(unittest.TestCase):
    def test_keypoints_at_the_maxima_scored_by_the_lower(self) -> None:
        simcc_x = np.zeros((1, NUM_KEYPOINTS, 8), np.float32)
        simcc_y = np.zeros((1, NUM_KEYPOINTS, 8), np.float32)
        simcc_x[0, :, 6] = 0.9
        simcc_y[0, :, 2] = 0.7
        keypoints, scores = decode_simcc(simcc_x, simcc_y)
        self.assertTrue(np.allclose(keypoints[0], [3.0, 1.0]))
        self.assertTrue(np.allclose(scores[0], 0.7))



class ModelFileTest(unittest.TestCase):
    def test_each_resolution_rtmpose_comes_in_has_its_file_and_the_others_none(self) -> None:
        settings = pose.PoseEstimatorSettings(model_type=ModelType.TRT, resolution=Resolution.HIGH)
        self.assertEqual(settings.model, 'rtmpose-l_384x288.trt')
        self.assertEqual((settings.width, settings.height), (288, 384))
        settings.resolution = Resolution.ULTRA
        with self.assertRaises(ValueError):
            _ = settings.model

if __name__ == '__main__':
    unittest.main()
