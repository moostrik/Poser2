"""DepthAligner: depth and its confidence moved onto the color frame's view, on plain CPU tensors with known geometry."""

import unittest

import numpy as np
import torch

from modules.camera import depth_projection
from modules.inference import DepthAligner, depth_meters


class DepthAlignerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.aligner = DepthAligner()
        self.size = (8, 6)
        self.to_ray, self.to_uv = depth_projection(None, self.size)      # depth already on the color's view

    def aligned(self, depth: np.ndarray, confidence: np.ndarray | None = None, flip_h: bool = False,
                homography: np.ndarray | None = None) -> np.ndarray:
        """The aligned (H, W, 2) to a tenth of a millimeter: the paths round the float's last bits each their own way."""
        source: torch.Tensor | None = None if confidence is None else torch.from_numpy(confidence.astype(np.uint8))
        return np.round(self.aligner.align(torch.from_numpy(depth.astype(np.float32)), source, self.to_ray, self.to_uv, homography,
                                           flip_h, self.size).numpy(), 4)

    def align(self, depth: np.ndarray, flip_h: bool = False, homography: np.ndarray | None = None) -> np.ndarray:
        """The aligned meters alone."""
        return self.aligned(depth, None, flip_h, homography)[..., 0]

    def test_millimeters_become_meters_and_none_stays_none(self) -> None:
        meters = depth_meters(torch.tensor([[0, 1500]], dtype=torch.int32))
        self.assertEqual(meters.tolist(), [[0.0, 1.5]])

    def test_a_depth_on_the_color_view_lands_where_it_was(self) -> None:
        depth = np.full((6, 8), 2.0)
        depth[2, 3] = 1.0
        out = self.align(depth)
        self.assertEqual(out[2, 3], 1.0)
        self.assertEqual(out[0, 0], 2.0)

    def test_no_depth_is_never_blended_and_where_nothing_lands_is_zero(self) -> None:
        depth = np.zeros((6, 8))
        depth[2, 3] = 1.5
        out = self.align(depth)
        self.assertEqual(sorted(set(out.ravel().tolist())), [0.0, 1.5])
        self.assertEqual(int((out > 0).sum()), 4)                   # the 2x2 cells around where it landed

    def test_the_nearest_is_kept_where_two_land_on_one_cell(self) -> None:
        depth = np.zeros((6, 8))
        depth[2, 3] = 3.0
        depth[2, 4] = 1.0                                           # its 2x2 overlaps the first's
        out = self.align(depth)                                     # each covers its own cell and the one right of it
        self.assertEqual(out[2, 3], 3.0)
        self.assertEqual(out[2, 4], 1.0)                            # the shared cell: the nearer one
        self.assertEqual(out[2, 5], 1.0)

    def test_the_mirror_turns_the_view(self) -> None:
        depth = np.zeros((6, 8))
        depth[2, 1] = 1.0
        out = self.align(depth, flip_h=True)
        self.assertEqual(out[2, 6], 1.0)
        self.assertEqual(out[2, 1], 0.0)

    def test_a_homography_s_scale_changes_the_depth_as_a_level_camera_would_read_it(self) -> None:
        depth = np.full((6, 8), 2.0)
        scale = np.diag([1.0, 1.0, 1.25])                            # third row: the depth scale of every pixel
        out = self.align(depth, homography=scale)
        self.assertAlmostEqual(float(out[0, 0]), 2.5, places=5)

    def test_the_nearest_brings_its_own_confidence(self) -> None:
        depth = np.zeros((6, 8))
        depth[2, 3], depth[2, 4] = 3.0, 1.0
        confidence = np.zeros((6, 8))
        confidence[2, 3], confidence[2, 4] = 255, 51
        out = self.aligned(depth, confidence)
        self.assertEqual(out[2, 3, 1], 1.0)                         # the far one's own cell
        self.assertEqual(out[2, 4, 1], 0.2)                         # the shared cell: the nearer one's, though less sure

    def test_of_two_equally_near_the_surer_wins(self) -> None:
        depth = np.zeros((6, 8))
        depth[2, 3], depth[2, 4] = 2.0, 2.0
        confidence = np.zeros((6, 8))
        confidence[2, 3], confidence[2, 4] = 102, 204
        out = self.aligned(depth, confidence)
        self.assertEqual(out[2, 4, 1], 0.8)
        self.assertEqual(out[2, 3, 1], 0.4)

    def test_where_nothing_lands_both_are_zero_and_a_frame_without_confidence_is_sure(self) -> None:
        depth = np.zeros((6, 8))
        depth[2, 3] = 1.5
        out = self.aligned(depth)
        self.assertEqual(out[0, 0].tolist(), [0.0, 0.0])
        self.assertEqual(out[2, 3].tolist(), [1.5, 1.0])

    def test_the_depth_keeps_a_tenth_of_a_millimeter_beside_its_confidence(self) -> None:
        depth = np.full((6, 8), 2.9871)
        out = self.aligned(depth, np.zeros((6, 8)))
        self.assertAlmostEqual(float(out[0, 0, 0]), 2.9871, delta=1e-4)
        self.assertEqual(float(out[0, 0, 1]), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class AlignStagedTest(DepthAlignerTest):
    """The same cases through the CUDA kernels, from uint16 millimeters and uint8 confidence in pinned buffers."""
    kernels: bool = True

    def aligned(self, depth: np.ndarray, confidence: np.ndarray | None = None, flip_h: bool = False,
                homography: np.ndarray | None = None) -> np.ndarray:
        staged = torch.from_numpy(np.round(depth * 1000.0).astype(np.uint16)).pin_memory()
        staged_confidence = None if confidence is None else torch.from_numpy(confidence.astype(np.uint8)).pin_memory()
        aligned = self.aligner.align_staged(staged, staged_confidence, self.to_ray, self.to_uv, homography, flip_h, self.size,
                                            kernels=self.kernels)
        torch.cuda.synchronize()
        return np.round(aligned.cpu().numpy(), 4)

    def test_the_depth_keeps_a_tenth_of_a_millimeter_beside_its_confidence(self) -> None:
        depth = np.full((6, 8), 2.987)                              # whole millimeters, as staged
        out = self.aligned(depth, np.zeros((6, 8)))
        self.assertAlmostEqual(float(out[0, 0, 0]), 2.987, delta=1e-4)
        self.assertEqual(float(out[0, 0, 1]), 0.0)

    def test_each_frame_is_a_tensor_of_its_own_and_the_cells_start_empty(self) -> None:
        first = self.align(np.full((6, 8), 2.0))
        second = self.align(np.zeros((6, 8)))
        self.assertEqual(float(first[0, 0]), 2.0)
        self.assertEqual(float(second.max()), 0.0)                  # nothing of the frame before

    def test_a_frame_without_confidence_after_one_with_is_sure(self) -> None:
        self.aligned(np.full((6, 8), 2.0), np.full((6, 8), 10))
        out = self.aligned(np.full((6, 8), 2.0))
        self.assertEqual(float(out[0, 0, 1]), 1.0)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class AlignStagedGraphTest(AlignStagedTest):
    """The same cases with the kernels off: `align` as a CUDA graph."""
    kernels = False


if __name__ == '__main__':
    unittest.main()
