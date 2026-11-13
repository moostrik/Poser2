"""`CameraColumns`: the band reduced on the GPU matches the CPU row of the same frame, and is read back without
waiting. Runs on a CUDA install."""

import time
import unittest

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import GpuImage, ImageUpload

from ..camera_columns import CameraColumns, SampleMethod


def _cpu_row(image: np.ndarray, band_top: float, band_bottom: float, method: SampleMethod) -> np.ndarray:
    """The row as the layer reduced the CPU frame: the band's rows, in [0, 1], reduced per column."""
    height: int = image.shape[0]
    band: np.ndarray = image[int(band_top * height):int(band_bottom * height)].astype(np.float32) / 255.0
    if method == SampleMethod.MAX:
        return band.max(axis=0)
    if method == SampleMethod.MIN:
        return band.min(axis=0)
    return band.mean(axis=0)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class CameraColumnsTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.frame_image: np.ndarray = rng.integers(0, 256, (120, 160), dtype=np.uint8)      # a mono camera's frame
        stream = torch.cuda.Stream()
        images, ready = ImageUpload().upload([CameraFrame(2, FrameType.VIDEO, self.frame_image, 1.0)], stream)
        self.image = GpuImage(2, 1.0, images[0], ready)

    def wait_for_row(self, columns: CameraColumns) -> np.ndarray:
        end: float = time.perf_counter() + 2.0
        while time.perf_counter() < end:
            row = columns.newest(2)
            if row is not None:
                return row
            time.sleep(0.001)
        self.fail("no row came")

    def test_each_method_matches_the_cpu_row(self) -> None:
        for method in SampleMethod:
            with self.subTest(method=method.name):
                columns = CameraColumns()
                columns.submit(self.image, 0.4, 0.6, method)
                row: np.ndarray = self.wait_for_row(columns)
                np.testing.assert_allclose(row, _cpu_row(self.frame_image, 0.4, 0.6, method), atol=2e-3)   # float16 upload

    def test_an_image_already_submitted_is_not_reduced_again(self) -> None:
        columns = CameraColumns()
        columns.submit(self.image, 0.4, 0.6, SampleMethod.MAX)
        first: np.ndarray = self.wait_for_row(columns)
        columns.submit(self.image, 0.0, 1.0, SampleMethod.MIN)
        torch.cuda.synchronize()
        self.assertIs(columns.newest(2), first)


if __name__ == '__main__':
    unittest.main()
