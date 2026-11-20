"""The segmenter's queue on its own thread, with a fake model: a batch replaced while the model runs is
counted and reaches no callback; the batches it ran reach the callbacks from its thread; every player's memory is
cleared each `reset_interval` batches."""

import threading
import unittest

import torch

from modules.inference import ModelType, crop, segmentation
from modules.pose import BBox, Frame
from modules.utils import Rect


class States(dict):
    """The segmenter's recurrent states, its clears counted."""
    resets: int = 0

    def clear(self) -> None:
        self.resets += 1
        super().clear()


class FakeModel:
    """The segmenter's model steps without a GPU: its `_infer`, blocking until released."""
    def __init__(self) -> None:
        self.running = threading.Event()
        self.release = threading.Event()

    def infer(self, images: list[torch.Tensor], track_ids: list[int],
              ready: torch.cuda.Event | None) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        self.running.set()
        self.release.wait(2.0)
        n = len(images)
        return torch.zeros((n, 1, 1)), torch.zeros((n, 3, 1, 1)), WRITTEN


WRITTEN = torch.cuda.Event()      # the model's event, recorded after its last step


def crop_image() -> crop.Image:
    return crop.Image(crop=torch.zeros(1))


class SegmenterTest(unittest.TestCase):
    def test_a_batch_replaced_while_the_model_runs_is_counted_not_run(self) -> None:
        settings = segmentation.SegmenterSettings(model_type=ModelType.ONNX, reset_interval=2)
        settings.initialize()
        segmenter = segmentation.Segmenter(settings)
        runner = FakeModel()
        states = States()
        segmenter._setup = lambda: None
        segmenter._infer = runner.infer
        segmenter._states = states
        told: list[tuple[list[int], str]] = []
        done = threading.Event()

        events: list[object] = []

        def on_masks(poses, masks) -> None:
            told.append((sorted(masks), threading.current_thread().name))
            events.extend(image.ready for image in masks.values())
            if len(told) == 2:
                done.set()

        segmenter.add_segmentation_callback(on_masks)
        segmenter.start()
        try:
            for _ in range(200):
                if segmenter.is_ready:
                    break
                threading.Event().wait(0.01)
            frame = Frame(track_id=0, cam_id=0, timestamp=1.0, features={BBox: BBox.from_rect(Rect(0.0, 0.0, 1.0, 1.0))})
            segmenter.submit({0: frame}, {0: crop_image()})           # runs
            self.assertTrue(runner.running.wait(2.0))
            segmenter.submit({1: frame}, {1: crop_image()})           # waits
            segmenter.submit({2: frame}, {2: crop_image()})           # replaces it
            runner.release.set()
            self.assertTrue(done.wait(2.0))
        finally:
            segmenter.stop()

        self.assertEqual(settings.dropped, 1)
        self.assertEqual([ids for ids, _ in told], [[0], [2]])
        self.assertTrue(all(name == 'Segmenter' for _, name in told))
        self.assertTrue(events and all(event is WRITTEN for event in events))    # every image carries the runner's event
        self.assertEqual(states.resets, 1)                     # at the second batch


class ModelFileTest(unittest.TestCase):
    def test_the_resolution_and_the_runtime_pick_the_file_each_in_its_own_folder(self) -> None:
        settings = segmentation.SegmenterSettings(resolution=segmentation.Resolution.HIGH, onnx_path='onnx_folder',
                                                  engine_path='engine_folder')
        self.assertEqual((settings.width, settings.height), (288, 384))
        self.assertEqual(settings.model_file(settings.model), 'engine_folder/rvm_mobilenetv3_384x288_fp16.trt')
        settings.model_type = ModelType.ONNX
        self.assertEqual(settings.model_file(settings.model), 'onnx_folder/rvm_mobilenetv3_384x288_fp16.onnx')


if __name__ == '__main__':
    unittest.main()
