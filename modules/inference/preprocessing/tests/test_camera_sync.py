"""`CameraSync`: the set reaches its callbacks in registration order, each camera's newest message in it."""

import random
import threading
import time
import unittest
from functools import partial

import numpy as np

from modules.camera import CameraFrame, FrameType

from .. import CameraSync, CameraSyncSettings, SyncMessage
from ..camera_sync import PHASE_FRAMES


class _Recorder:
    """Appends its name to a shared list each time the sync fires."""

    def __init__(self, name: int, calls: list[int]) -> None:
        self._name = name
        self._calls = calls

    def fire(self, items: dict[int, SyncMessage]) -> None:
        self._calls.append(self._name)
        self.items = items


def frame(cam_id: int, timestamp: float) -> CameraFrame:
    return CameraFrame(cam_id, FrameType.VIDEO, np.zeros((2, 2), np.uint8), timestamp)


def synced(fps: float) -> tuple[CameraSync, list[dict[int, SyncMessage]]]:
    """A sync, not started, and the sets it hands on: the tests decide its messages on their own thread (`_decide`)."""
    settings = CameraSyncSettings()
    settings.fps = fps
    sync = CameraSync(settings)
    sets: list[dict[int, SyncMessage]] = []
    sync.add_set_callback(sets.append)
    return sync, sets


class CameraSyncTest(unittest.TestCase):
    def test_callbacks_run_in_registration_order(self) -> None:
        sync = CameraSync(CameraSyncSettings())
        calls: list[int] = []
        recorders = [_Recorder(i, calls) for i in range(16)]
        for i, recorder in enumerate(recorders):
            sync.add_set_callback(recorder.fire if i % 2 else partial(_Recorder.fire, recorder))
        sync.add_set_callback(recorders[1].fire)                 # registered twice: called once, first position
        first: CameraFrame = frame(0, 12.5)
        sync._decide(first)                                      # one camera: its first frame syncs
        self.assertEqual(calls, list(range(16)))
        self.assertEqual(recorders[0].items, {0: first})         # the set carries the frame itself

    def test_a_depth_frame_does_not_sync(self) -> None:
        sync = CameraSync(CameraSyncSettings())
        calls: list[int] = []
        sync.add_set_callback(_Recorder(0, calls).fire)
        sync.submit(CameraFrame(0, FrameType.DEPTH, np.zeros((2, 2), np.uint16), 12.5))
        self.assertEqual(calls, [])

    def test_a_set_holds_one_frame_of_each_camera(self) -> None:
        sync, sets = synced(10.0)
        for moment in range(6):
            t: float = 1.0 + moment * 0.1
            for cam_id, offset in ((0, 0.0), (1, 0.01), (2, 0.05)):
                sync._decide(frame(cam_id, t + offset))
        last: dict[int, SyncMessage] = sets[-1]
        self.assertEqual(sorted(last), [0, 1, 2])
        self.assertAlmostEqual(last[0].timestamp, 1.5)
        self.assertAlmostEqual(last[1].timestamp, 1.51)
        self.assertAlmostEqual(last[2].timestamp, 1.55)

    def test_a_camera_that_stops_is_left_out_after_two_frames(self) -> None:
        sync, sets = synced(10.0)
        sync._decide(frame(1, 0.9))                              # camera 1 sends once, then stops
        for moment in range(5):
            sync._decide(frame(0, 1.0 + moment * 0.1))
        self.assertEqual(sorted(sets[-1]), [0])

    def test_any_arrival_order_gives_one_set_per_moment_and_every_frame_once(self) -> None:
        sync, sets = synced(23.0)
        rng = random.Random(1)
        frames = [frame(cam_id, moment / 23.0 + rng.uniform(0.0, 0.030)) for moment in range(230) for cam_id in range(4)]
        frames.sort(key=lambda f: f.timestamp)                  # each moment's four come in any order
        for f in frames:
            sync._decide(f)
        sent: list[SyncMessage] = [m for items in sets for m in items.values()]
        self.assertEqual(len(sent), len(frames) - len(sync._open))          # every frame once, the open set aside
        self.assertEqual(len({id(m) for m in sent}), len(sent))
        self.assertAlmostEqual(len(sets) / 230, 1.0, delta=0.02)

    def test_a_camera_a_frame_behind_joins_the_moment_it_is_closest_to(self) -> None:
        sync, sets = synced(23.0)
        period: float = 1.0 / 23.0
        for moment in range(40):                                # camera 3 runs 40 ms behind the others
            for cam_id in range(4):
                sync._decide(frame(cam_id, moment * period + (0.040 if cam_id == 3 else 0.0)))
        spreads: list[float] = [max(m.timestamp for m in s.values()) - min(m.timestamp for m in s.values())
                                for s in sets[PHASE_FRAMES:] if len(s) == 4]
        self.assertTrue(spreads)
        self.assertLess(max(spreads), 0.005)                    # with the next moment's frames, 3.5 ms away

    def test_a_lost_frame_closes_its_set_at_the_next_moment_without_it(self) -> None:
        sync, sets = synced(23.0)
        for moment in range(30):
            for cam_id in range(4):
                if not (moment == 20 and cam_id == 2):
                    sync._decide(frame(cam_id, moment / 23.0 + cam_id * 0.002))
        self.assertIn([0, 1, 3], [sorted(s) for s in sets])
        self.assertEqual(sum(len(s) for s in sets) + len(sync._open), 30 * 4 - 1)

    def test_a_message_before_start_is_an_error_and_one_after_stop_is_dropped(self) -> None:
        sync, sets = synced(10.0)
        with self.assertRaises(RuntimeError):
            sync.submit(frame(0, 1.0))
        sync.start()
        sync.stop()
        sync.submit(frame(0, 1.1))
        self.assertEqual(sets, [])


class CameraSyncThreadTest(unittest.TestCase):
    """Once started, the syncs are decided on the sync's own thread; a submit only queues."""

    def setUp(self) -> None:
        settings = CameraSyncSettings()
        settings.fps = 10.0
        self.sync = CameraSync(settings, stream_name='frame_sync')
        self.fired: list[tuple[float, str]] = []
        self.sync.add_set_callback(lambda items: self.fired.append((items[0].timestamp, threading.current_thread().name)))
        self.sync.start()

    def tearDown(self) -> None:
        self.sync.stop()

    def wait_for(self, count: int) -> None:
        end: float = time.perf_counter() + 1.0
        while len(self.fired) < count and time.perf_counter() < end:
            time.sleep(0.001)

    def test_a_sync_fires_on_the_sync_thread(self) -> None:
        self.sync.submit(frame(0, 1.0))
        self.wait_for(1)
        self.assertEqual(self.fired, [(1.0, 'CameraSync frame_sync')])

    def test_beyond_sixteen_waiting_a_cameras_oldest_is_dropped_and_counted(self) -> None:
        release: threading.Event = threading.Event()
        held: threading.Event = threading.Event()

        def hold(items: dict[int, SyncMessage]) -> None:
            if not held.is_set():
                held.set()
                release.wait(2.0)
        self.sync.add_set_callback(hold)
        self.sync.submit(frame(0, 1.0))
        self.assertTrue(held.wait(1.0))                     # the sync's thread is held: what comes now waits
        for i in range(1, 19):
            self.sync.submit(frame(0, 1.0 + i * 0.1))     # eighteen frames, sixteen kept
        release.set()
        self.wait_for(17)
        self.assertEqual(self.sync.dropped, 2)
        self.assertEqual(self.fired[1][0], 1.3)             # the frames at 1.1 and 1.2 were dropped


if __name__ == '__main__':
    unittest.main()
