"""`CameraSync`: the sync reaches its callbacks in registration order."""

import threading
import time
import unittest
from functools import partial

import numpy as np

from modules.camera import CameraFrame, FrameType
from modules.inference import CameraSync, CameraSyncSettings


class _Recorder:
    """Appends its name to a shared list each time the sync fires."""

    def __init__(self, name: int, calls: list[int]) -> None:
        self._name = name
        self._calls = calls

    def fire(self, timestamp: float) -> None:
        self._calls.append(self._name)
        self.timestamp = timestamp


class CameraSyncTest(unittest.TestCase):
    def test_callbacks_run_in_registration_order(self) -> None:
        sync = CameraSync(CameraSyncSettings())
        calls: list[int] = []
        recorders = [_Recorder(i, calls) for i in range(16)]
        for i, recorder in enumerate(recorders):
            sync.add_sync_callback(recorder.fire if i % 2 else partial(_Recorder.fire, recorder))
        sync.add_sync_callback(recorders[1].fire)                 # registered twice: called once, first position
        sync.submit_frame(CameraFrame(0, FrameType.VIDEO, np.zeros((2, 2), np.uint8), 12.5))   # one camera: its first frame syncs
        self.assertEqual(calls, list(range(16)))
        self.assertEqual(recorders[0].timestamp, 12.5)       # the sync carries the frame's time

    def test_a_depth_frame_does_not_sync(self) -> None:
        sync = CameraSync(CameraSyncSettings())
        calls: list[int] = []
        sync.add_sync_callback(_Recorder(0, calls).fire)
        sync.submit_frame(CameraFrame(0, FrameType.DEPTH, np.zeros((2, 2), np.uint16), 12.5))
        self.assertEqual(calls, [])


def frame(cam_id: int, timestamp: float) -> CameraFrame:
    return CameraFrame(cam_id, FrameType.VIDEO, np.zeros((2, 2), np.uint8), timestamp)


class CameraSyncThreadTest(unittest.TestCase):
    """Once started, the syncs are decided on the sync's own thread; a submit only queues."""

    def setUp(self) -> None:
        settings = CameraSyncSettings()
        settings.fps = 10.0
        self.sync = CameraSync(settings, stream_name='frame_sync')
        self.fired: list[tuple[float, str]] = []
        self.sync.add_sync_callback(lambda timestamp: self.fired.append((timestamp, threading.current_thread().name)))
        self.sync.start()

    def tearDown(self) -> None:
        self.sync.stop()

    def wait_for(self, count: int) -> None:
        end: float = time.perf_counter() + 1.0
        while len(self.fired) < count and time.perf_counter() < end:
            time.sleep(0.001)

    def test_a_sync_fires_on_the_sync_thread(self) -> None:
        self.sync.submit_frame(frame(0, 1.0))
        self.wait_for(1)
        self.assertEqual(self.fired, [(1.0, 'CameraSync frame_sync')])

    def test_beyond_sixteen_waiting_a_cameras_oldest_is_dropped_and_counted(self) -> None:
        release: threading.Event = threading.Event()
        held: threading.Event = threading.Event()

        def hold(timestamp: float) -> None:
            if not held.is_set():
                held.set()
                release.wait(2.0)
        self.sync.add_sync_callback(hold)
        self.sync.submit_frame(frame(0, 1.0))
        self.assertTrue(held.wait(1.0))                     # the sync's thread is held: what comes now waits
        for i in range(1, 19):
            self.sync.submit_frame(frame(0, 1.0 + i * 0.1))     # eighteen frames, sixteen kept
        release.set()
        self.wait_for(17)
        self.assertEqual(self.sync.dropped, 2)
        self.assertEqual(self.fired[1][0], 1.3)             # the frames at 1.1 and 1.2 were dropped


if __name__ == '__main__':
    unittest.main()
