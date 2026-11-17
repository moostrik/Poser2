"""A runner's waiting batch replaced by a newer one: counted on the model's settings, and its callbacks told on the
submitting thread, without the model loaded."""

import threading
import unittest

from .. import optical_flow, segmentation
from ..optical_flow.io import OpticalFlowInput
from ..optical_flow.runner_onnx import RunnerONNX as FlowRunner
from ..segmentation.io import SegmentationInput
from ..segmentation.runner_onnx import RunnerONNX as SegmentationRunner


class RunnerDropTest(unittest.TestCase):
    def check(self, runner, settings, first, second, add_callback) -> None:
        told: list[tuple[int, bool, str]] = []
        add_callback(lambda output: told.append((output.batch_id, output.processed, threading.current_thread().name)))
        runner._model_ready.set()                       # as if loaded; the thread is never started
        runner.submit(first)
        runner.submit(second)                           # replaces the first before it ran
        self.assertEqual(told, [(1, False, threading.current_thread().name)])
        self.assertEqual(settings.dropped, 1)

    def test_optical_flow(self) -> None:
        settings = optical_flow.Settings()
        settings.initialize()
        runner = FlowRunner(settings)
        self.check(runner, settings, OpticalFlowInput(batch_id=1, gpu_image_pairs=[], track_ids=[]),
                   OpticalFlowInput(batch_id=2, gpu_image_pairs=[], track_ids=[]), runner.add_flow_callback)

    def test_segmentation(self) -> None:
        settings = segmentation.Settings()
        settings.initialize()
        runner = SegmentationRunner(settings)
        self.check(runner, settings, SegmentationInput(batch_id=1, gpu_images=[], track_ids=[]),
                   SegmentationInput(batch_id=2, gpu_images=[], track_ids=[]), runner.add_segmentation_callback)


if __name__ == '__main__':
    unittest.main()
