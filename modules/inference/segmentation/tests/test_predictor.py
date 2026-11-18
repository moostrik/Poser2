"""The segmentation predictor's queue on its own thread, with a fake runner: a batch replaced while the model runs is
counted and reaches no callback; the batches it ran reach the callbacks from its thread; every player's memory is
cleared each `reset_interval` batches."""

import threading
import unittest

from modules.inference import ModelType, segmentation
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
        self.resets: int = 0

    def setup(self) -> None:
        pass

    def reset_states(self) -> None:
        self.resets += 1

    def infer(self, gpu_images, track_ids, ready):
        self.running.set()
        self.release.wait(2.0)
        n = len(gpu_images)
        return [f'mask{i}' for i in range(n)], [f'fgr{i}' for i in range(n)]


class Crop:
    crop = object()
    ready = None


class PredictorTest(unittest.TestCase):
    def test_a_batch_replaced_while_the_model_runs_is_counted_not_run(self) -> None:
        settings = segmentation.SegmentationPredictorSettings(model_type=ModelType.ONNX, reset_interval=2)
        settings.initialize()
        predictor = segmentation.Predictor(settings)
        runner = FakeRunner()
        predictor._runner = runner
        told: list[tuple[list[int], str]] = []
        done = threading.Event()

        def on_masks(poses, masks) -> None:
            told.append((sorted(masks), threading.current_thread().name))
            if len(told) == 2:
                done.set()

        predictor.add_segmentation_callback(on_masks)
        predictor.start()
        try:
            for _ in range(200):
                if predictor.is_ready:
                    break
                threading.Event().wait(0.01)
            frame = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.0, 0.0, 1.0, 1.0))})
            predictor.submit({0: frame}, {0: Crop()})           # runs
            self.assertTrue(runner.running.wait(2.0))
            predictor.submit({1: frame}, {1: Crop()})           # waits
            predictor.submit({2: frame}, {2: Crop()})           # replaces it
            runner.release.set()
            self.assertTrue(done.wait(2.0))
        finally:
            predictor.stop()

        self.assertEqual(settings.dropped, 1)
        self.assertEqual([ids for ids, _ in told], [[0], [2]])
        self.assertTrue(all(name == 'SegmentationPredictor' for _, name in told))
        self.assertEqual(runner.resets, 1)                     # at the second batch


if __name__ == '__main__':
    unittest.main()
