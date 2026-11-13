"""RecordingSync on its own thread: the submits return at once, and its thread matches in the order the parts came."""

import threading
import time
import unittest

from modules.camera import FrameType
from modules.recording import RecordedCapture, RecordingSync
from modules.recording.tests.recordings import cam_frame, cam_tracklets

FRAMES: int = 12           # under `KEPT`: one sender may run that far ahead of the other
INTERVAL: float = 0.02      # s between captures, 50 fps


class RecordingSyncThreadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sync = RecordingSync(with_depth=False)
        self.captures: list[RecordedCapture] = []
        self.threads: list[str] = []

        def on_capture(capture: RecordedCapture) -> None:
            self.captures.append(capture)
            self.threads.append(threading.current_thread().name)
        self.sync.add_capture_callback(on_capture)
        self.sync.expect_parts(0, tracklets=True, poses=False)
        self.sync.start()

    def tearDown(self) -> None:
        self.sync.stop()

    def test_parts_from_two_threads_match_on_the_sync_thread_in_order(self) -> None:
        def frames() -> None:
            for i in range(FRAMES):
                self.sync.submit_frame(cam_frame(0, i, i * INTERVAL))

        def tracklets() -> None:
            for i in range(FRAMES):
                self.sync.submit_tracklets(cam_tracklets(0, i * INTERVAL))
        senders = [threading.Thread(target=frames), threading.Thread(target=tracklets)]
        for sender in senders:
            sender.start()
        for sender in senders:
            sender.join()
        self.sync.settle()
        self.sync.flush()

        self.assertEqual([c.video.timestamp for c in self.captures], [i * INTERVAL for i in range(FRAMES)])
        self.assertTrue(all(c.tracklets is not None for c in self.captures))
        self.assertEqual(set(self.threads), {'RecordingSync'})
        self.assertEqual(self.sync.missed.total('tracklets'), 0)

    def test_a_submit_returns_before_its_capture_is_handed_on(self) -> None:
        handed: threading.Event = threading.Event()
        release: threading.Event = threading.Event()

        def slow(capture: RecordedCapture) -> None:
            handed.set()
            release.wait(2.0)
        self.sync.add_capture_callback(slow)
        self.sync.submit_frame(cam_frame(0, 0, 0.0))
        self.sync.submit_tracklets(cam_tracklets(0, 0.0))
        self.assertTrue(handed.wait(2.0))
        start: float = time.perf_counter()
        self.sync.submit_frame(cam_frame(0, 1, INTERVAL))         # the sync's thread is held in the callback
        self.assertLess(time.perf_counter() - start, 0.05)
        self.assertGreater(self.sync.waiting, 0)
        release.set()
        self.sync.settle()

    def test_flush_hands_on_what_waits_and_returns_after(self) -> None:
        self.sync.submit_frame(cam_frame(0, 0, 0.0))                # its tracklets never come
        self.sync.flush()
        self.assertEqual(len(self.captures), 1)
        self.assertIsNone(self.captures[0].tracklets)
        self.assertEqual(self.sync.waiting, 0)

    def test_depth_is_left_out_when_not_recorded(self) -> None:
        frame = cam_frame(0, 0, 0.0)
        depth = type(frame)(0, FrameType.DEPTH, frame.image[..., 0].astype('uint16'), 0.0)
        self.sync.submit_frame(depth)
        self.sync.settle()
        self.assertEqual(self.sync.waiting, 0)


class RecordingSyncInlineTest(unittest.TestCase):
    def test_without_its_thread_a_part_is_matched_on_the_caller(self) -> None:
        sync = RecordingSync(with_depth=False)
        captures: list[RecordedCapture] = []
        sync.add_capture_callback(captures.append)
        sync.submit_frame(cam_frame(0, 0, 0.0))
        self.assertEqual(len(captures), 1)


if __name__ == '__main__':
    unittest.main()
