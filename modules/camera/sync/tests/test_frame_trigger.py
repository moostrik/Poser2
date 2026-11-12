"""`FrameTrigger`: the trigger reaches its callbacks in registration order."""

import unittest
from functools import partial

import numpy as np

from modules.camera import CameraFrame, FrameType, FrameTrigger, FrameTriggerSettings


class _Recorder:
    """Appends its name to a shared list each time the trigger fires."""

    def __init__(self, name: int, calls: list[int]) -> None:
        self._name = name
        self._calls = calls

    def fire(self, timestamp: float) -> None:
        self._calls.append(self._name)
        self.timestamp = timestamp


class FrameTriggerTest(unittest.TestCase):
    def test_callbacks_run_in_registration_order(self) -> None:
        trigger = FrameTrigger(FrameTriggerSettings())
        calls: list[int] = []
        recorders = [_Recorder(i, calls) for i in range(16)]
        for i, recorder in enumerate(recorders):
            trigger.add_trigger_callback(recorder.fire if i % 2 else partial(_Recorder.fire, recorder))
        trigger.add_trigger_callback(recorders[1].fire)                 # registered twice: called once, first position
        trigger.submit_frame(CameraFrame(0, FrameType.VIDEO, np.zeros((2, 2), np.uint8), 12.5))   # one camera: its first frame triggers
        self.assertEqual(calls, list(range(16)))
        self.assertEqual(recorders[0].timestamp, 12.5)       # the trigger carries the frame's time

    def test_a_depth_frame_does_not_trigger(self) -> None:
        trigger = FrameTrigger(FrameTriggerSettings())
        calls: list[int] = []
        trigger.add_trigger_callback(_Recorder(0, calls).fire)
        trigger.submit_frame(CameraFrame(0, FrameType.DEPTH, np.zeros((2, 2), np.uint16), 12.5))
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
