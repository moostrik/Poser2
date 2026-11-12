"""`PlayClock`: the recorded time of a play, one catch-up for two late readers, carried from chunk to chunk."""

import tempfile
import unittest

import numpy as np

from modules.camera import FrameMetadata, FrameType, Player, PlayerSettings
from modules.camera.player import PlayClock


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


class HandOverTest(unittest.TestCase):
    def setUp(self) -> None:
        settings = PlayerSettings()
        settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
        settings.initialize()
        self.folder = tempfile.TemporaryDirectory()
        self.player = Player(settings, self.folder.name)
        self.player._set_play_chunk(0)
        self.player.play_types = {FrameType.VIDEO, FrameType.DEPTH}
        self.out: list[tuple[int, FrameType]] = []
        self.player.frame_callbacks.append(lambda cam, kind, image, metadata, captured: self.out.append((int(image[0]), kind)))

    def tearDown(self) -> None:
        self.folder.cleanup()

    def push(self, frame_id: int, kind: FrameType) -> None:
        self.player._frame_sync_callback(0, kind, np.array([frame_id]), 0, frame_id)

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
        self.assertEqual(self.player._unpaired, 1)


if __name__ == '__main__':
    unittest.main()
