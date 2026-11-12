"""`PlayClock`: the recorded time of a play, one catch-up for two late readers, carried from chunk to chunk.
`FramePairing`: a frame's streams handed over together, in order. `Timeline`: the markers and places on a
recording's timeline, without a video."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.camera import FrameMetadata, FrameType, PlayClock, RecordingFolder, Timeline
from modules.camera.frame_pairing import FramePairing


def frames(times: list[float], latency: float = 0.03) -> list[FrameMetadata]:
    return [FrameMetadata(t, video_latency=latency) for t in times]


class PlayClockTest(unittest.TestCase):
    def test_a_frame_comes_at_its_recorded_time_plus_its_latency(self) -> None:
        clock = PlayClock({0: frames([10.0, 10.02, 10.06])}, 0)
        clock.start(100.0)
        self.assertAlmostEqual(clock.capture(0, 2) or 0.0, 100.06)
        self.assertAlmostEqual(clock.moment(0, 2) or 0.0, 100.09)

    def test_two_readers_late_for_one_stall_move_it_once(self) -> None:
        clock = PlayClock({0: frames([10.0, 10.02])}, 0)
        clock.start(100.0)
        first = clock.catch_up(0, 1, 100.5)         # the color's reader: due at 100.05, 0.45 s late
        second = clock.catch_up(0, 1, 100.5)        # the depth's reader, late for the same stall
        self.assertAlmostEqual(first, 100.5)
        self.assertAlmostEqual(second, 100.5)
        self.assertAlmostEqual(clock.moment(0, 1) or 0.0, 100.5)

    def test_the_next_chunk_runs_on_from_the_one_before(self) -> None:
        first = PlayClock({0: frames([10.0, 10.02])}, 0)
        first.start(100.0)
        second = PlayClock({0: frames([10.04, 10.06])}, 0)
        second.carry_on(first, 200.0)
        self.assertAlmostEqual(second.capture(0, 0) or 0.0, 100.04)     # 20 ms after the last, as recorded

    def test_a_chunk_without_times_starts_anew(self) -> None:
        first = PlayClock({0: frames([10.0])}, 0)
        first.start(100.0)
        second = PlayClock({}, 0)
        second.carry_on(first, 200.0)
        self.assertIsNone(second.capture(0, 0))


class FakeReaders:
    """A chunk's readers as the pairing sees them: its number, streams, metadata and clock."""
    def __init__(self) -> None:
        self.chunk = 0
        self.types = {FrameType.VIDEO, FrameType.DEPTH}
        self.clock = PlayClock({}, 0)

    def frame(self, cam_id: int, frame_id: int) -> None:
        return None


class FramePairingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pairing = FramePairing(1, [FrameType.VIDEO, FrameType.DEPTH])
        self.pairing.start(FakeReaders())   # pyright: ignore[reportArgumentType]
        self.out: list[tuple[int, FrameType]] = []
        self.pairing.frame_callbacks.append(lambda cam, kind, image, metadata, captured: self.out.append((int(image[0]), kind)))

    def push(self, frame_id: int, kind: FrameType, chunk: int = 0) -> None:
        self.pairing.submit_frame(0, kind, np.array([frame_id]), chunk, frame_id)

    def test_a_frame_goes_out_with_both_its_streams(self) -> None:
        self.push(4, FrameType.VIDEO)
        self.assertEqual(self.out, [])
        self.push(4, FrameType.DEPTH)
        self.assertEqual(self.out, [(4, FrameType.VIDEO), (4, FrameType.DEPTH)])

    def test_a_frame_older_than_one_gone_out_is_dropped_and_counted(self) -> None:
        self.push(4, FrameType.VIDEO)
        self.push(5, FrameType.VIDEO)
        self.push(5, FrameType.DEPTH)       # 4 never gets its depth
        self.push(4, FrameType.DEPTH)       # too late: 5 went out
        self.assertEqual(self.out, [(5, FrameType.VIDEO), (5, FrameType.DEPTH)])
        self.assertEqual(self.pairing.take_unpaired(), 1)
        self.assertEqual(self.pairing.take_unpaired(), 0)

    def test_another_chunk_s_frames_are_not_handed_over(self) -> None:
        self.push(4, FrameType.VIDEO, chunk=1)
        self.push(4, FrameType.DEPTH, chunk=1)
        self.assertEqual(self.out, [])

    def test_stopped_nothing_is_handed_over(self) -> None:
        self.pairing.start(None)
        self.push(4, FrameType.VIDEO)
        self.push(4, FrameType.DEPTH)
        self.assertEqual(self.out, [])


class TimelineTest(unittest.TestCase):
    def test_the_markers_fall_in_their_chunks(self) -> None:
        timeline = Timeline.of([100, 50], [0.5, 0.9])     # frames 75 to 135 of 150
        self.assertEqual((timeline.start_chunk, timeline.start_offset, timeline.end_chunk, timeline.end_offset), (0, 75, 1, 35))

    def test_markers_in_either_order_and_at_the_ends(self) -> None:
        self.assertEqual(Timeline.of([100, 50], [1.0, 0.0]), Timeline([100, 50], 0, 0, 1, 50))

    def test_a_place_and_whether_it_is_within_the_markers(self) -> None:
        timeline = Timeline.of([100, 50], [0.5, 0.9])
        self.assertEqual(timeline.place(0.2), 30)
        self.assertFalse(timeline.within_markers(30))
        self.assertTrue(timeline.within_markers(timeline.place(0.6)))
        self.assertEqual(timeline.global_frame(1, 10), 110)

    def test_no_frames_no_markers(self) -> None:
        self.assertEqual(Timeline.of([], [0.2, 0.8]), Timeline([]))


class RecordingFolderTest(unittest.TestCase):
    def test_a_folder_without_chunks_is_no_recording(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'empty').mkdir()
            (Path(root) / 'clip').mkdir()
            (Path(root) / 'clip' / '0_002.mkv').write_bytes(b'')
            folders = RecordingFolder.find(Path(root))
            self.assertEqual(list(folders), ['clip'])
            self.assertEqual(folders['clip'].chunks, 2)


if __name__ == '__main__':
    unittest.main()
