"""The detector thread end to end on a stub model: a graph with the YOLOv8 output shape that
reports the same person in every image, so the batching on the GPU, the decoding and the callbacks
are checked without the real model (ONNX Runtime's CUDA provider)."""

import os
import tempfile
import time
import unittest

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from modules.inference import GpuImage, ModelType
from modules.inference.detection import PersonDetector, Resolution, Settings
from modules.utils import Rect

WIDTH, HEIGHT = 512, 448
MODEL_FILE = 'yolov8n_coco_512x448.onnx'


def write_stub_model(path: str, symbolic: bool = False) -> None:
    """input (N, 3, 448, 512) -> output (N, 84, 2): candidate 0 a person at the middle with score
    0.9 (cx 256, cy 224, w 128, h 224), candidate 1 a person with score 0.2. `symbolic` gives the
    input the shape ultralytics exports, (N, 3, height, width)."""
    rows = np.zeros((1, 84, 2), dtype=np.float32)
    rows[0, 0:5, 0] = (256, 224, 128, 224, 0.9)
    rows[0, 0:5, 1] = (100, 100, 50, 50, 0.2)
    nodes = [
        helper.make_node('Shape', ['images'], ['shape']),
        helper.make_node('Gather', ['shape', 'zero'], ['batch'], axis=0),
        helper.make_node('Concat', ['batch', 'tail'], ['out_shape'], axis=0),
        helper.make_node('Expand', ['rows', 'out_shape'], ['output0']),
    ]
    graph = helper.make_graph(
        nodes, 'stub_yolo',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['N', 3, 'height', 'width'] if symbolic else ['N', 3, HEIGHT, WIDTH])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, ['N', 84, 2])],
        initializer=[
            numpy_helper.from_array(rows, 'rows'),
            numpy_helper.from_array(np.array([0], dtype=np.int64), 'zero'),
            numpy_helper.from_array(np.array([84, 2], dtype=np.int64), 'tail'),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, path)


def uploaded(cam_id: int, height: int, width: int, timestamp: float) -> GpuImage:
    """An image as the `Uploader` hands it on: float16 RGB CHW on the GPU, with its written event."""
    image: torch.Tensor = torch.zeros((3, height, width), dtype=torch.float16, device='cuda')
    ready = torch.cuda.Event()
    ready.record()
    return GpuImage(cam_id, timestamp, image, ready)


def wait_for(condition, timeout: float = 5.0) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class PersonDetectorTest(unittest.TestCase):
    symbolic: bool = False

    def setUp(self) -> None:
        self.folder = tempfile.mkdtemp()
        write_stub_model(os.path.join(self.folder, MODEL_FILE), self.symbolic)
        self.settings = Settings(model_path=self.folder)
        self.settings.initialize()
        self.detector = PersonDetector(self.settings)
        self.detections = []
        self.detector.add_detection_callback(self.detections.append)
        self.detector.start()
        self.assertTrue(wait_for(lambda: self.detector.is_ready))

    def tearDown(self) -> None:
        self.detector.stop()

    def test_every_submitted_image_gets_its_detection_with_boxes_normalised_to_the_frame_and_its_time(self) -> None:
        self.detector.submit(uploaded(0, 600, 800, 1.5))       # another size
        self.detector.submit(uploaded(2, 300, 400, 2.5))
        self.assertTrue(wait_for(lambda: len(self.detections) == 2))
        by_cam = {d.cam_id: d for d in self.detections}
        self.assertEqual(set(by_cam), {0, 2})
        self.assertEqual((by_cam[0].timestamp, by_cam[2].timestamp), (1.5, 2.5))
        for d in by_cam.values():
            self.assertEqual(len(d.boxes), 1)                                   # the 0.2 one is under the threshold
            self.assertEqual(d.boxes[0], Rect(0.375, 0.25, 0.25, 0.5))
            self.assertAlmostEqual(d.scores[0], 0.9, places=5)

    def test_the_batch_is_the_images_at_the_model_s_size_lying_in_memory_as_its_shape_says(self) -> None:
        """The runners take the buffer's address, so a tensor that is only a view of another order
        reaches the model scrambled."""
        batch = self.detector._batch([uploaded(0, 600, 800, 0.0), uploaded(1, 720, 896, 0.0)])
        torch.cuda.synchronize()
        self.assertEqual(tuple(batch.shape), (2, 3, HEIGHT, WIDTH))
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(batch.is_contiguous())

    def test_the_image_is_stretched_whole_into_the_model(self) -> None:
        image = uploaded(0, 600, 800, 0.0)
        image.image[0, :, :400] = 1.0                           # red on the left half
        batch = self.detector._batch([image])
        torch.cuda.synchronize()
        red = batch[0, 0].cpu().numpy()
        self.assertTrue(np.allclose(red[:, :WIDTH // 2 - 1], 1.0) and np.allclose(red[:, WIDTH // 2 + 1:], 0.0))

    def test_the_threshold_is_read_at_use(self) -> None:
        self.settings.confidence = 0.1
        self.detector.submit(uploaded(0, 448, 512, 0.0))
        self.assertTrue(wait_for(lambda: len(self.detections) == 1))
        self.assertEqual(len(self.detections[0].boxes), 2)


class SymbolicInputTest(PersonDetectorTest):
    """The shape ultralytics exports: height and width as symbols."""
    symbolic = True


class ModelFileTest(unittest.TestCase):
    def test_the_size_and_the_runtime_pick_the_file_each_in_its_own_folder(self) -> None:
        settings = Settings(resolution=Resolution.R640X512, model_path='onnx_folder', engine_path='engine_folder')
        self.assertEqual((settings.width, settings.height), (640, 512))
        self.assertEqual(settings.model_file, 'onnx_folder/yolov8n_coco_640x512.onnx')
        settings.model_type = ModelType.TRT
        self.assertEqual(settings.model_file, 'engine_folder/yolov8n_coco_640x512.trt')


class MissingModelTest(unittest.TestCase):
    def test_a_missing_file_fails_to_load_with_its_name(self) -> None:
        settings = Settings(model_path=tempfile.mkdtemp())
        settings.initialize()
        detector = PersonDetector(settings)
        with self.assertLogs('modules.inference.detection.person_detector', level='ERROR') as logs:
            detector.start()
            detector.join(timeout=5.0)
        self.assertFalse(detector.is_ready)
        self.assertIn(MODEL_FILE, logs.output[0])


if __name__ == '__main__':
    unittest.main()
