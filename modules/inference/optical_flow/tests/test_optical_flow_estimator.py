"""The optical flow estimator's queue on its own thread, with a fake model: a batch replaced while the model runs is
counted and reaches no callback; the batches it ran reach the callbacks from its thread; a pose without a previous
crop is left out."""

import threading
import unittest

from modules.inference import ModelType, optical_flow
from modules.pose import BBox, Frame
from modules.utils import Rect


class FakeModel:
    """The estimator's model steps without a GPU: its `_infer`, blocking until released."""
    def __init__(self) -> None:
        self.running = threading.Event()
        self.release = threading.Event()

    def infer(self, pairs, ready):
        self.running.set()
        self.release.wait(2.0)
        return [f'flow{i}' for i in range(len(pairs))]


class Crop:
    def __init__(self, prev_crop: object | None = object()) -> None:
        self.crop = object()
        self.prev_crop = prev_crop
        self.ready = None


class OpticalFlowEstimatorTest(unittest.TestCase):
    def test_a_batch_replaced_while_the_model_runs_is_counted_not_run(self) -> None:
        settings = optical_flow.OpticalFlowEstimatorSettings(model_type=ModelType.ONNX)
        settings.initialize()
        estimator = optical_flow.OpticalFlowEstimator(settings)
        runner = FakeModel()
        estimator._setup = lambda: None
        estimator._infer = runner.infer
        told: list[tuple[list[int], str]] = []
        done = threading.Event()

        def on_flows(flows) -> None:
            told.append((sorted(flows), threading.current_thread().name))
            if len(told) == 2:
                done.set()

        estimator.add_flow_callback(on_flows)
        estimator.start()
        try:
            for _ in range(200):
                if estimator.is_ready:
                    break
                threading.Event().wait(0.01)
            frame = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.0, 0.0, 1.0, 1.0))})
            estimator.submit({0: frame}, {0: Crop()})                               # runs
            self.assertTrue(runner.running.wait(2.0))
            estimator.submit({1: frame}, {1: Crop()})                               # waits
            estimator.submit({2: frame, 3: frame}, {2: Crop(), 3: Crop(None)})      # replaces it; 3 has no previous crop
            runner.release.set()
            self.assertTrue(done.wait(2.0))
        finally:
            estimator.stop()

        self.assertEqual(settings.dropped, 1)
        self.assertEqual([ids for ids, _ in told], [[0], [2]])
        self.assertTrue(all(name == 'OpticalFlowEstimator' for _, name in told))


if __name__ == '__main__':
    unittest.main()
