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
        return self.aligner.align(torch.from_numpy(depth.astype(np.float32)), self.to_ray, self.to_uv, homography, flip_h, self.size).numpy()

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
class AlignStagedTest(unittest.TestCase):
    def test_the_graph_gives_what_align_gives_frame_after_frame_and_after_a_new_homography(self) -> None:
        aligner = DepthAligner()
        size = (8, 6)
        to_ray, to_uv = depth_projection(None, size)
        rng = np.random.default_rng(1)
        homographies = [None, None, np.diag([1.0, 1.0, 1.25])]
        results = []
        for homography in homographies:
            frame = rng.integers(0, 3000, (6, 8)).astype(np.uint16)
            staged = torch.from_numpy(frame).pin_memory()
            got = aligner.align_staged(staged, to_ray, to_uv, homography, False, size)
            want = aligner.align(depth_metres(torch.from_numpy(frame).cuda()), to_ray, to_uv, homography, False, size)
            torch.cuda.synchronize()
            results.append(got)
            np.testing.assert_allclose(got.cpu().numpy(), want.cpu().numpy(), rtol=1e-6)
        self.assertIsNot(results[0], results[1])                    # each frame a tensor of its own


if __name__ == '__main__':
    unittest.main()
