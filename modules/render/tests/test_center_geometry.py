"""`CenterGeometry` keeps the output's proportions whatever the camera frame's shape."""

import math
import unittest
from typing import cast

from modules.board import HasFrames
from modules.gl import Texture
from modules.pose import BBox, Frame
from modules.render.layers.center.center_geometry import CenterGeometry, CenterGeometrySettings, fixed_window
from modules.utils import Point2f, Rect


OUTPUT_ASPECT: float = 9.0 / 16.0


class CalculateRoiTest(unittest.TestCase):
    def test_a_square_frame_gets_the_plain_crop(self) -> None:
        rotation, distance, roi = CenterGeometry._calculate_roi(
            Point2f(0.5, 0.3), Point2f(0.5, 0.6), Point2f(0.5, 0.3), Point2f(0.5, 0.33), 0.27, OUTPUT_ASPECT)
        self.assertAlmostEqual(rotation, 0.0)
        self.assertAlmostEqual(distance, 0.3)
        self.assertAlmostEqual(roi.height, 0.3 / 0.27)
        self.assertAlmostEqual(roi.width, 0.3 / 0.27 * OUTPUT_ASPECT)

    def test_a_wider_frame_keeps_the_output_aspect_in_pixels(self) -> None:
        cam_aspect: float = 5.0 / 4.0
        _, _, roi = CenterGeometry._calculate_roi(
            Point2f(0.5, 0.3), Point2f(0.5, 0.6), Point2f(0.5, 0.3), Point2f(0.5, 0.33), 0.27, OUTPUT_ASPECT, cam_aspect)
        self.assertAlmostEqual(roi.width * cam_aspect / roi.height, OUTPUT_ASPECT)

    def test_a_lean_is_measured_in_true_proportions(self) -> None:
        cam_aspect: float = 5.0 / 4.0
        rotation, distance, _ = CenterGeometry._calculate_roi(
            Point2f(0.5, 0.3), Point2f(0.5 + 0.1 / cam_aspect, 0.4), Point2f(0.5, 0.3), Point2f(0.5, 0.33), 0.27, OUTPUT_ASPECT, cam_aspect)
        self.assertAlmostEqual(rotation, math.pi / 4.0)
        self.assertAlmostEqual(distance, math.hypot(0.1, 0.1))


class FakeTexture:
    """The camera texture's size, all the geometry reads of it."""
    width: int = 896
    height: int = 720
    allocated: bool = True


class FakeBoard:
    def __init__(self) -> None:
        self.pose: Frame | None = None

    def get_frame(self, stage: int, track_id: int) -> Frame | None:
        return self.pose


def pose_at(x: float) -> Frame:
    return Frame(track_id=0, cam_id=0, timestamp=x, features={BBox: BBox.from_rect(Rect(x, 0.2, 0.3, 0.6))})


class FixedViewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = CenterGeometrySettings()
        self.settings.follow_pose = False
        self.board = FakeBoard()
        self.geometry = CenterGeometry(0, cast(HasFrames, self.board), cast(Texture, FakeTexture()), self.settings)

    def test_the_view_stays_put_while_the_pose_moves_and_is_there_without_one(self) -> None:
        self.geometry.update()
        self.assertTrue(self.geometry.ready)
        without = self.geometry.image_geometry.crop_roi
        for x in (0.1, 0.5):
            self.board.pose = pose_at(x)
            self.geometry.update()
            self.assertEqual(self.geometry.image_geometry.crop_roi, without)
            self.assertEqual(self.geometry.image_geometry.rotation, 0.0)
        self.assertFalse(self.geometry.lost)

    def test_following_the_pose_needs_one(self) -> None:
        self.settings.follow_pose = True
        self.geometry.update()
        self.assertFalse(self.geometry.ready)


class DepthViewTest(unittest.TestCase):
    """The view's share of the camera's field of view, and the depth's sample step in the view."""

    def setUp(self) -> None:
        self.settings = CenterGeometrySettings()
        self.settings.follow_pose = False
        self.settings.fixed_height = 0.5
        self.geometry = CenterGeometry(0, cast(HasFrames, FakeBoard()), cast(Texture, FakeTexture()), self.settings, fov=90.0)
        self.geometry.update()

    def test_the_span_is_the_crops_share_of_the_image_span(self) -> None:
        crop = self.geometry.image_geometry.crop_roi
        self.assertAlmostEqual(self.geometry.view_span, crop.width * 2.0)      # 90°: the image spans 2 m per metre of depth

    def test_the_depth_step_is_a_sample_over_the_crop(self) -> None:
        crop = self.geometry.image_geometry.crop_roi
        step = self.geometry.depth_step((1.0 / 320.0, 1.0 / 200.0))
        self.assertAlmostEqual(step[0], 1.0 / 320.0 / crop.width)
        self.assertAlmostEqual(step[1], 1.0 / 200.0 / crop.height)

    def test_no_field_of_view_spans_nothing(self) -> None:
        geometry = CenterGeometry(0, cast(HasFrames, FakeBoard()), cast(Texture, FakeTexture()), self.settings)
        geometry.update()
        self.assertEqual(geometry.view_span, 0.0)


class FixedWindowTest(unittest.TestCase):
    def test_a_full_height_window_is_centered_with_the_output_aspect_in_pixels(self) -> None:
        cam_aspect: float = 896.0 / 720.0
        window = fixed_window(0.5, 0.5, 1.0, OUTPUT_ASPECT, cam_aspect)
        self.assertAlmostEqual(window.y, 0.0)
        self.assertAlmostEqual(window.height, 1.0)
        self.assertAlmostEqual(window.x + window.width / 2.0, 0.5)
        self.assertAlmostEqual(window.width * cam_aspect / window.height, OUTPUT_ASPECT)

    def test_the_window_sits_where_its_center_says(self) -> None:
        window = fixed_window(0.3, 0.6, 0.5, OUTPUT_ASPECT, 1.0)
        self.assertAlmostEqual(window.x + window.width / 2.0, 0.3)
        self.assertAlmostEqual(window.y + window.height / 2.0, 0.6)
        self.assertAlmostEqual(window.height, 0.5)


if __name__ == '__main__':
    unittest.main()
