"""Tests for Sync: the bang reaches its callbacks in registration order."""

import unittest
from functools import partial

import numpy as np

from modules.camera import CameraFrame, FrameType, Sync, SyncSettings


class _Recorder:
    """Appends its name to a shared list each time the bang arrives."""

    def __init__(self, name: int, calls: list[int]) -> None:
        self._name = name
        self._calls = calls

    def bang(self, timestamp: float) -> None:
        self._calls.append(self._name)
        self.timestamp = timestamp


class SyncCallbackTest(unittest.TestCase):
    def test_callbacks_run_in_registration_order(self) -> None:
        sync = Sync(SyncSettings())
        calls: list[int] = []
        recorders = [_Recorder(i, calls) for i in range(16)]
        for i, recorder in enumerate(recorders):
            sync.add_sync_callback(recorder.bang if i % 2 else partial(_Recorder.bang, recorder))
        sync.add_sync_callback(recorders[1].bang)                 # registered twice: called once, first position
        sync.submit_frame(CameraFrame(0, FrameType.VIDEO, np.zeros((2, 2), np.uint8), 12.5))   # one camera: its first frame bangs
        self.assertEqual(calls, list(range(16)))
        self.assertEqual(recorders[0].timestamp, 12.5)       # the bang carries the frame's time

    def test_a_depth_frame_does_not_bang(self) -> None:
        sync = Sync(SyncSettings())
        calls: list[int] = []
        sync.add_sync_callback(_Recorder(0, calls).bang)
        sync.submit_frame(CameraFrame(0, FrameType.DEPTH, np.zeros((2, 2), np.uint16), 12.5))
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
