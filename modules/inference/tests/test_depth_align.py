"""DepthAligner: depth moved onto the color frame's view, on plain CPU tensors with known geometry."""

import unittest

import numpy as np
import torch

from modules.camera import depth_projection
from modules.inference import DepthAligner, depth_metres


class DepthAlignerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.aligner = DepthAligner()
        self.size = (8, 6)
        self.to_ray, self.to_uv = depth_projection(None, self.size)      # depth already on the color's view

    def align(self, depth: np.ndarray, flip_h: bool = False, homography: np.ndarray | None = None) -> np.ndarray:
        """The aligned depth to a tenth of a millimetre: the paths round the float's last bits each their own way."""
        return np.round(self.aligner.align(torch.from_numpy(depth.astype(np.float32)), self.to_ray, self.to_uv, homography, flip_h, self.size).numpy(), 4)

    def test_millimetres_become_metres_and_none_stays_none(self) -> None:
        metres = depth_metres(torch.tensor([[0, 1500]], dtype=torch.int32))
        self.assertEqual(metres.tolist(), [[0.0, 1.5]])

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


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class AlignStagedTest(DepthAlignerTest):
    """The same cases through the CUDA kernels, from uint16 millimetres in a pinned buffer."""
    kernels: bool = True

    def align(self, depth: np.ndarray, flip_h: bool = False, homography: np.ndarray | None = None) -> np.ndarray:
        staged = torch.from_numpy(np.round(depth * 1000.0).astype(np.uint16)).pin_memory()
        aligned = self.aligner.align_staged(staged, self.to_ray, self.to_uv, homography, flip_h, self.size, kernels=self.kernels)
        torch.cuda.synchronize()
        return np.round(aligned.cpu().numpy(), 4)

    def test_each_frame_is_a_tensor_of_its_own_and_the_cells_start_empty(self) -> None:
        first = self.align(np.full((6, 8), 2.0))
        second = self.align(np.zeros((6, 8)))
        self.assertEqual(float(first[0, 0]), 2.0)
        self.assertEqual(float(second.max()), 0.0)                  # nothing of the frame before


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class AlignStagedGraphTest(AlignStagedTest):
    """The same cases with the kernels off: `align` as a CUDA graph."""
    kernels = False


if __name__ == '__main__':
    unittest.main()
