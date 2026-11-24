"""The OAK4's drops on the panel: a frame the keypoint join drops and a depth frame before the frames' lens is known
each count in the camera's readings. Runs without a device. On a depthai 3 install (`.venv`)."""

import unittest
from unittest.mock import MagicMock

import depthai as dai

from modules.camera.oak4 import ModelChain, Oak4CameraSettings
from modules.camera.oak4.camera import Camera
from modules.camera.oak4.pipeline import DEPTH_STREAM, CONFIDENCE_STREAM


def depth_group() -> MagicMock:
    """A depth message group as the depth queue hands it over, captured now."""
    frame = MagicMock()
    frame.getTimestamp.return_value = dai.Clock.now()
    return MagicMock(**{'__getitem__.side_effect': lambda name: {DEPTH_STREAM: frame, CONFIDENCE_STREAM: frame}[name]})


class DropReadingsTest(unittest.TestCase):
    def test_a_frame_the_join_drops_shows_in_the_readings(self) -> None:
        camera = Camera(Oak4CameraSettings(model_chain=ModelChain.TWO_MODELS))
        assert camera.keypoint_join is not None
        camera.keypoint_join.put_keypoints(1, [], [])
        camera.keypoint_join.put_keypoints(1 + camera.keypoint_join.keep + 1, [], [])
        self.assertEqual(camera.readings.dropped_tracklets, 1)

    def test_a_depth_frame_before_the_lens_is_known_shows_in_the_readings(self) -> None:
        camera = Camera(Oak4CameraSettings())
        camera._depth_callback(DEPTH_STREAM, depth_group())
        camera._depth_callback(DEPTH_STREAM, depth_group())
        self.assertEqual(camera.readings.dropped_depth, 2)

    def test_the_counts_stay_over_a_close(self) -> None:
        camera = Camera(Oak4CameraSettings())
        camera._depth_callback(DEPTH_STREAM, depth_group())
        camera._close()
        self.assertEqual(camera.readings.dropped_depth, 1)


if __name__ == '__main__':
    unittest.main()
