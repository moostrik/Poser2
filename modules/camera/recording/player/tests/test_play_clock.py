"""`PlayClock`: the recorded time of a play, one catch-up for two late readers, carried from chunk
to chunk."""

import unittest

from modules.camera import FrameMetadata, PlayClock


def frames(times: list[float], latency: float = 0.03) -> list[FrameMetadata]:
    return [FrameMetadata(t, video_latency=latency) for t in times]


class PlayClockTest(unittest.TestCase):
    def test_a_frame_comes_at_its_recorded_time_plus_its_latency(self) -> None:
        clock = PlayClock({0: frames([10.0, 10.02, 10.06])}, {0: 0})
        clock.start(100.0)
        self.assertAlmostEqual(clock.capture(0, 2) or 0.0, 100.06)
        self.assertAlmostEqual(clock.moment(0, 2) or 0.0, 100.09)

    def test_two_readers_late_for_one_stall_move_it_once(self) -> None:
        clock = PlayClock({0: frames([10.0, 10.02])}, {0: 0})
        clock.start(100.0)
        first = clock.catch_up(0, 1, 100.5)         # the color's reader: due at 100.05, 0.45 s late
        second = clock.catch_up(0, 1, 100.5)        # the depth's reader, late for the same stall
        self.assertAlmostEqual(first, 100.5)
        self.assertAlmostEqual(second, 100.5)
        self.assertAlmostEqual(clock.moment(0, 1) or 0.0, 100.5)

    def test_the_next_chunk_runs_on_from_the_one_before(self) -> None:
        first = PlayClock({0: frames([10.0, 10.02])}, {0: 0})
        first.start(100.0)
        second = PlayClock({0: frames([10.04, 10.06])}, {0: 0})
        second.carry_on(first, 200.0)
        self.assertAlmostEqual(second.capture(0, 0) or 0.0, 100.04)     # 20 ms after the last, as recorded

    def test_a_chunk_without_times_starts_anew(self) -> None:
        first = PlayClock({0: frames([10.0])}, {0: 0})
        first.start(100.0)
        second = PlayClock({}, {})
        second.carry_on(first, 200.0)
        self.assertIsNone(second.capture(0, 0))

    def test_cameras_start_at_the_earliest_first_frame(self) -> None:
        clock = PlayClock({0: frames([10.0, 10.02, 10.04]), 1: frames([10.01, 10.03])}, {0: 1, 1: 0})
        clock.start(100.0)
        self.assertAlmostEqual(clock.capture(1, 0) or 0.0, 100.0)       # camera 1's 10.01 is the base
        self.assertAlmostEqual(clock.capture(0, 1) or 0.0, 100.01)

    def test_a_catch_up_in_the_next_chunk_moves_the_one_ending_too(self) -> None:
        first = PlayClock({0: frames([10.0, 10.02])}, {0: 0})
        first.start(100.0)
        second = PlayClock({0: frames([10.04])}, {0: 0})
        second.carry_on(first, 200.0)
        second.catch_up(0, 0, 100.17)          # due at 100.07, 0.1 s late
        self.assertAlmostEqual(first.capture(0, 1) or 0.0, 100.12)


if __name__ == '__main__':
    unittest.main()
