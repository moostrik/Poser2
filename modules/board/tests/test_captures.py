"""The board's captures: kept in order until the render takes them, the oldest dropped and counted beyond the bound."""

import unittest
from typing import cast

from modules.board import CaptureStoreMixin
from modules.board.captures import KEPT
from modules.inference import Capture


class _Capture:
    def __init__(self, cam_id: int, timestamp: float) -> None:
        self.cam_id = cam_id
        self.timestamp = timestamp


def capture(timestamp: float, cam_id: int = 0) -> Capture:
    return cast(Capture, _Capture(cam_id, timestamp))


class CaptureStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = CaptureStoreMixin()

    def test_captures_are_taken_in_order_and_once(self) -> None:
        for t in (1.0, 2.0):
            self.board.set_capture(capture(t))
        self.assertEqual([c.timestamp for c in self.board.take_captures(0)], [1.0, 2.0])
        self.assertEqual(self.board.take_captures(0), [])

    def test_cameras_are_kept_apart(self) -> None:
        self.board.set_capture(capture(1.0, cam_id=1))
        self.assertEqual(self.board.take_captures(0), [])
        self.assertEqual([c.timestamp for c in self.board.take_captures(1)], [1.0])

    def test_beyond_the_bound_the_oldest_is_dropped_and_counted(self) -> None:
        for t in range(KEPT + 2):
            self.board.set_capture(capture(float(t)))
        self.assertEqual([c.timestamp for c in self.board.take_captures(0)], [float(t) for t in range(2, KEPT + 2)])
        self.assertEqual(self.board.get_dropped_captures(), 2)


if __name__ == '__main__':
    unittest.main()
