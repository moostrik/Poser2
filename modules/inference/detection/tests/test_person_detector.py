"""The detector thread end to end on a stub model: a graph with the YOLOv8 output shape that
reports the same person in every image, so the batching, the decoding and the callbacks are
checked without the real model or a GPU (ONNX Runtime's CPU provider)."""

import logging
import os
import tempfile
import time
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from modules.inference import ModelType
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
        logging.disable(logging.WARNING)        # the CPU-provider warning
        self.detector.start()
        self.assertTrue(wait_for(lambda: self.detector.is_ready))

    def tearDown(self) -> None:
        self.detector.stop()
        logging.disable(logging.NOTSET)

    def test_every_submitted_frame_gets_its_detection_with_boxes_normalised_to_the_frame(self) -> None:
        self.detector.submit(0, np.zeros((600, 800, 3), dtype=np.uint8))       # colour, another size
        self.detector.submit(2, np.zeros((300, 400), dtype=np.uint8))          # gray
        self.assertTrue(wait_for(lambda: len(self.detections) == 2))
        by_cam = {d.cam_id: d for d in self.detections}
        self.assertEqual(set(by_cam), {0, 2})
        for d in by_cam.values():
            self.assertEqual(len(d.boxes), 1)                                   # the 0.2 one is under the threshold
            self.assertEqual(d.boxes[0], Rect(0.375, 0.25, 0.25, 0.5))
            self.assertAlmostEqual(d.scores[0], 0.9, places=5)

    def test_the_input_lies_in_memory_as_its_shape_says(self) -> None:
        """TensorRT takes the buffer's address, so an array that is only a view of another order
        reaches the model scrambled."""
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        batch = np.stack([self.detector._preprocess(image), self.detector._preprocess(image)])
        self.assertTrue(batch.flags['C_CONTIGUOUS'])

    def test_the_threshold_is_read_at_use(self) -> None:
        self.settings.confidence = 0.1
        self.detector.submit(0, np.zeros((448, 512, 3), dtype=np.uint8))
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
