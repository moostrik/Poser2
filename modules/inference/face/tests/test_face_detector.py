"""The face detector on the GPU against the CPU path it replaces: `data/face.jpg` (smile's warm-up photo, fitted to
512 x 640) and the values the CPU path (insightface's preprocessing with cv2) gave on it, `data/face_expected.json`.
On ONNX Runtime, and on TensorRT where smile's engines are built (`scripts/makemodels.bat smile`)."""

import json
import os
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from modules.camera import CameraFrame, FrameType

from ... import GpuImage, ImageUpload, ModelType
from .. import FaceDetector, FaceDetectorSettings, FaceFrame
from ..face_detector import landmark_matrix

DATA: Path = Path(__file__).parent / 'data'
ENGINES: str = 'apps/smile/data/models'


def detector(model_type: ModelType) -> FaceDetector:
    settings = FaceDetectorSettings()
    settings.model_type = model_type
    settings.engine_path = ENGINES
    settings.initialize()
    return FaceDetector(settings)


def uploaded(image: np.ndarray, timestamp: float = 1.0) -> GpuImage:
    frame = CameraFrame(0, FrameType.VIDEO, image, timestamp)
    images, ready = ImageUpload().upload([frame], torch.cuda.Stream())
    return GpuImage(0, timestamp, images[0], ready, frame)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class FaceDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        image = cv2.imread(str(DATA / 'face.jpg'))
        assert image is not None, 'face.jpg is missing'
        cls.image: np.ndarray = image
        cls.expected = json.loads((DATA / 'face_expected.json').read_text())['faces']

    def check(self, frame: FaceFrame) -> None:
        """The frame's face against the CPU path's."""
        expected = self.expected[0]
        face = frame.faces[0]
        np.testing.assert_allclose(face.bbox, expected['bbox'], atol=1.0)
        np.testing.assert_allclose(face.kps, expected['kps'], atol=1.0)
        np.testing.assert_allclose(face.landmarks, expected['landmarks'], atol=1.5)
        self.assertAlmostEqual(face.score, expected['score'], delta=0.01)
        self.assertAlmostEqual(face.pose.yaw, expected['yaw'], delta=0.5)
        self.assertAlmostEqual(face.pose.pitch, expected['pitch'], delta=0.5)
        self.assertAlmostEqual(face.pose.roll, expected['roll'], delta=0.5)
        self.assertAlmostEqual(face.eyes, expected['eyes'], delta=0.01)

    def run_on(self, model_type: ModelType) -> None:
        faces = detector(model_type)
        frame: FaceFrame = faces.detect(uploaded(self.image))
        self.assertEqual(len(frame.faces), len(self.expected))
        self.check(frame)
        self.assertEqual((frame.width, frame.height), (512, 640))
        self.assertIsNotNone(frame.image.frame)
        assert frame.image.frame is not None
        self.assertIs(frame.image.frame.image, self.image)                 # the CPU frame of the moment comes along
        # two faces side by side: both found, and the landmarks of both in one batch as of each alone
        pair: np.ndarray = np.ascontiguousarray(np.hstack([self.image, self.image]))
        image: GpuImage = uploaded(pair)
        frame = faces.detect(image)
        self.assertEqual(len(frame.faces), 2)
        pixels: torch.Tensor = image.image.unsqueeze(0).float() * 255.0
        matrices = [landmark_matrix(face.bbox, face.pose.roll) for face in frame.faces]
        together = faces._landmark_points(pixels, matrices, 1024, 640)
        for matrix, points in zip(matrices, together):
            np.testing.assert_allclose(faces._landmark_points(pixels, [matrix], 1024, 640)[0], points, atol=0.05)

    def test_on_onnx_runtime_it_finds_what_the_cpu_path_found(self) -> None:
        self.run_on(ModelType.ONNX)

    @unittest.skipUnless(os.path.isfile(f'{ENGINES}/scrfd_10g_bnkps.trt') and os.path.isfile(f'{ENGINES}/2d106det.trt'),
                         'needs the face engines (scripts/makemodels.bat smile)')
    def test_on_tensorrt_it_finds_what_the_cpu_path_found(self) -> None:
        self.run_on(ModelType.TRT)

    def test_a_newer_image_replaces_one_waiting(self) -> None:
        faces = detector(ModelType.ONNX)
        faces.submit(uploaded(self.image, 1.0))
        faces.submit(uploaded(self.image, 2.0))
        self.assertEqual(faces.settings.dropped, 1)
        frames: list[FaceFrame] = []
        faces.add_face_callback(frames.append)
        faces.start()
        try:
            end = time.perf_counter() + 30.0
            while not frames and time.perf_counter() < end:
                time.sleep(0.05)
        finally:
            faces.stop()
        self.assertEqual([frame.timestamp for frame in frames], [2.0])


class ModelFileTest(unittest.TestCase):
    def test_each_model_and_the_runtime_pick_the_file_each_in_its_own_folder(self) -> None:
        settings = FaceDetectorSettings(onnx_path='onnx_folder', engine_path='engine_folder')
        self.assertEqual(settings.model_file('scrfd_10g_bnkps'), 'onnx_folder/scrfd_10g_bnkps.onnx')
        settings.model_type = ModelType.TRT
        self.assertEqual(settings.model_file('2d106det'), 'engine_folder/2d106det.trt')


if __name__ == '__main__':
    unittest.main()
