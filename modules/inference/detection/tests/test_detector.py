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

from modules.inference.detection import Detector, Settings
from modules.utils import Rect

WIDTH, HEIGHT = 512, 448


def write_stub_model(path: str) -> None:
    """input (N, 3, 448, 512) -> output (N, 84, 2): candidate 0 a person at the middle with score
    0.9 (cx 256, cy 224, w 128, h 224), candidate 1 a person with score 0.2."""
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
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['N', 3, HEIGHT, WIDTH])],
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


class DetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.mkdtemp()
        write_stub_model(os.path.join(self.folder, 'stub.onnx'))
        self.settings = Settings(model='stub.onnx', model_path=self.folder)
        self.settings.initialize()
        self.detector = Detector(self.settings)
        self.detections = []
        self.detector.register_callback(self.detections.append)
        logging.disable(logging.WARNING)        # the CPU-provider warning
        self.detector.start()
        self.assertTrue(wait_for(lambda: self.detector.is_ready))

    def tearDown(self) -> None:
        self.detector.stop()
        logging.disable(logging.NOTSET)

    def test_reads_the_input_size_off_the_model(self) -> None:
        self.assertEqual((self.detector.width, self.detector.height), (WIDTH, HEIGHT))

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

    def test_the_threshold_is_read_at_use(self) -> None:
        self.settings.confidence = 0.1
        self.detector.submit(0, np.zeros((448, 512, 3), dtype=np.uint8))
        self.assertTrue(wait_for(lambda: len(self.detections) == 1))
        self.assertEqual(len(self.detections[0].boxes), 2)


if __name__ == '__main__':
    unittest.main()
