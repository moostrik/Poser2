"""The board's frames: kept in order until the render takes them, the oldest dropped and counted beyond the bound."""

import unittest
from typing import cast

from modules.inference import InferenceFrame

from .. import InferenceFrameStoreMixin
from ..inference_frames import KEPT


class _Frame:
    def __init__(self, cam_id: int, timestamp: float) -> None:
        self.cam_id = cam_id
        self.timestamp = timestamp


def inference_frame(timestamp: float, cam_id: int = 0) -> InferenceFrame:
    return cast(InferenceFrame, _Frame(cam_id, timestamp))


class InferenceFrameStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = InferenceFrameStoreMixin()

    def test_frames_are_taken_in_order_and_once(self) -> None:
        for t in (1.0, 2.0):
            self.board.set_inference_frame(inference_frame(t))
        self.assertEqual([c.timestamp for c in self.board.take_inference_frames(0)], [1.0, 2.0])
        self.assertEqual(self.board.take_inference_frames(0), [])

    def test_cameras_are_kept_apart(self) -> None:
        self.board.set_inference_frame(inference_frame(1.0, cam_id=1))
        self.assertEqual(self.board.take_inference_frames(0), [])
        self.assertEqual([c.timestamp for c in self.board.take_inference_frames(1)], [1.0])

    def test_beyond_the_bound_the_oldest_is_dropped_and_counted(self) -> None:
        for t in range(KEPT + 2):
            self.board.set_inference_frame(inference_frame(float(t)))
        self.assertEqual([c.timestamp for c in self.board.take_inference_frames(0)], [float(t) for t in range(2, KEPT + 2)])
        self.assertEqual(self.board.get_dropped_inference_frames(), 2)


if __name__ == '__main__':
    unittest.main()
