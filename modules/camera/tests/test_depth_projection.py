"""`depth_projection`: where a depth camera's pixel lands in the video frame, as the GPU puts it there."""

import unittest

import numpy as np

from modules.camera import DepthAlignment, aligned_depth_size, depth_projection, mirrored_alignment

DEPTH_SIZE: tuple[int, int] = (480, 300)
DEPTH_LENS: list[list[float]] = [[212.4, 0.0, 240.6], [0.0, 212.4, 149.4], [0.0, 0.0, 1.0]]
COLOR_SIZE: tuple[int, int] = (896, 720)
COLOR_LENS: list[list[float]] = [[538.1, 0.0, 449.3], [0.0, 538.0, 369.8], [0.0, 0.0, 1.0]]
MOVE: np.ndarray = np.array([[1.0, 0.0032, -0.0077, -0.0376], [-0.0033, 0.9999, -0.016, -0.0001], [0.0077, 0.016, 0.9998, 0.0004], [0.0, 0.0, 0.0, 1.0]])
ALIGNMENT: DepthAlignment = DepthAlignment(DEPTH_LENS, COLOR_LENS, COLOR_SIZE, MOVE.tolist())


def land(alignment: DepthAlignment | None, size: tuple[int, int], pixel: tuple[float, float], metres: float) -> tuple[np.ndarray, float]:
    """A depth pixel through the projection, as the shader does: its uv and depth in the video frame."""
    to_ray, to_uv = depth_projection(alignment, size)
    video: np.ndarray = to_uv @ np.append(metres * (to_ray @ np.array([pixel[0], pixel[1], 1.0])), 1.0)
    return video[:2] / video[2], float(video[2])


class DepthProjectionTest(unittest.TestCase):
    def test_a_point_lands_where_the_video_camera_sees_it(self) -> None:
        point: np.ndarray = np.array([0.3, -0.2, 1.5])
        seen: np.ndarray = np.array(DEPTH_LENS) @ point
        uv, depth = land(ALIGNMENT, DEPTH_SIZE, (seen[0] / seen[2], seen[1] / seen[2]), point[2])
        in_video: np.ndarray = MOVE[:3, :3] @ point + MOVE[:3, 3]
        pixel: np.ndarray = np.array(COLOR_LENS) @ in_video
        np.testing.assert_array_almost_equal(uv, (pixel[:2] / pixel[2] + 0.5) / np.array(COLOR_SIZE))
        self.assertAlmostEqual(depth, in_video[2])

    def test_near_lands_further_aside_than_far(self) -> None:
        near, _ = land(ALIGNMENT, DEPTH_SIZE, (240.0, 150.0), 0.8)
        far, _ = land(ALIGNMENT, DEPTH_SIZE, (240.0, 150.0), 4.0)
        self.assertLess(near[0], far[0])        # the video camera sits to the right: near moves left more

    def test_without_an_alignment_a_pixel_keeps_its_place(self) -> None:
        uv, depth = land(None, (374, 300), (10.0, 20.0), 2.0)
        np.testing.assert_array_almost_equal(uv, (10.5 / 374, 20.5 / 300))
        self.assertAlmostEqual(depth, 2.0)

    def test_mirrored_depth_lands_mirrored(self) -> None:
        mirrored: DepthAlignment = mirrored_alignment(ALIGNMENT, DEPTH_SIZE[0])
        uv, depth = land(ALIGNMENT, DEPTH_SIZE, (100.0, 80.0), 1.2)
        mirrored_uv, mirrored_depth = land(mirrored, DEPTH_SIZE, (DEPTH_SIZE[0] - 1 - 100.0, 80.0), 1.2)
        np.testing.assert_array_almost_equal(mirrored_uv, (1.0 - uv[0], uv[1]))
        self.assertAlmostEqual(mirrored_depth, depth)

    def test_the_aligned_size_has_the_rows_and_the_video_aspect(self) -> None:
        self.assertEqual(aligned_depth_size(DEPTH_SIZE, ALIGNMENT), (374, 300))
        self.assertEqual(aligned_depth_size((374, 300), None), (374, 300))


if __name__ == '__main__':
    unittest.main()
