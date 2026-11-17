"""The optical-flow predictor's queue on its own thread, with a fake runner: a batch replaced while the model runs is
counted and reaches no callback; the batches it ran reach the callbacks from its thread; a pose without a previous
crop is left out."""

import threading
import unittest

from modules.inference import ModelType, optical_flow
from modules.pose import BBox, Frame
from modules.utils import Rect


class FakeRunner:
    model_file = 'fake'
    model_width = 192
    model_height = 256
    model_precision = 'FP32'
    max_batch = 4

    def __init__(self) -> None:
        self.running = threading.Event()
        self.release = threading.Event()

    def setup(self) -> None:
        pass

    def infer(self, pairs):
        self.running.set()
        self.release.wait(2.0)
        return [f'flow{i}' for i in range(len(pairs))], 0.0


class Crop:
    def __init__(self, prev_crop: object | None = object()) -> None:
        self.crop = object()
        self.prev_crop = prev_crop
        self.ready = None


class PredictorTest(unittest.TestCase):
    def test_a_batch_replaced_while_the_model_runs_is_counted_not_run(self) -> None:
        settings = optical_flow.OpticalFlowPredictorSettings(model_type=ModelType.ONNX)
        settings.initialize()
        predictor = optical_flow.Predictor(settings)
        runner = FakeRunner()
        predictor._runner = runner
        told: list[tuple[list[int], str]] = []
        done = threading.Event()

        def on_flows(flows) -> None:
            told.append((sorted(flows), threading.current_thread().name))
            if len(told) == 2:
                done.set()

        predictor.add_flow_callback(on_flows)
        predictor.start()
        try:
            for _ in range(200):
                if predictor.is_ready:
                    break
                threading.Event().wait(0.01)
            frame = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.0, 0.0, 1.0, 1.0))})
            predictor.submit({0: frame}, {0: Crop()})                               # runs
            self.assertTrue(runner.running.wait(2.0))
            predictor.submit({1: frame}, {1: Crop()})                               # waits
            predictor.submit({2: frame, 3: frame}, {2: Crop(), 3: Crop(None)})      # replaces it; 3 has no previous crop
            runner.release.set()
            self.assertTrue(done.wait(2.0))
        finally:
            predictor.stop()

        self.assertEqual(settings.dropped, 1)
        self.assertEqual([ids for ids, _ in told], [[0], [2]])
        self.assertTrue(all(name == 'OpticalFlowPredictor' for _, name in told))


if __name__ == '__main__':
    unittest.main()
