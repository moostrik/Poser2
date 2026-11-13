"""The recorder's pose hand-over: the app's poses go in as they are and are packed on the sync's thread."""

import threading
import unittest

from .. import FramePoses, RecordingSync
from .recordings import cam_frame, poses


class PosePackingTest(unittest.TestCase):
    def test_poses_are_packed_on_the_sync_thread_with_their_arrival(self) -> None:
        packed: list[tuple[object, float, str]] = []

        def pack(frames: object, arrival: float) -> list[FramePoses]:
            packed.append((frames, arrival, threading.current_thread().name))
            return []

        sync = RecordingSync(with_depth=False, pack_poses=pack)
        sync.start()
        try:
            frames = {3: 'a pose frame'}
            sync.submit_poses(frames, 2.5)
            sync.flush()                                # returns once every part queued before it is taken
            self.assertEqual(packed, [(frames, 2.5, 'RecordingSync')])
        finally:
            sync.stop()

    def test_packed_poses_reach_the_capture(self) -> None:
        sync = RecordingSync(with_depth=False, pack_poses=lambda frames, arrival: [poses(1.0)])
        sync.expect_parts(0, tracklets=False, poses=True)
        captures = []
        sync.add_capture_callback(captures.append)
        sync.submit_frame(cam_frame(0, 0, 1.0))
        sync.expect_poses(0, 1.0, True)
        sync.submit_poses({}, 1.05)
        self.assertEqual(len(captures), 1)
        self.assertIsNotNone(captures[0].poses)


if __name__ == '__main__':
    unittest.main()
