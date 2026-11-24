"""`Player`'s panel readouts: the playhead and the chunk at the frame shown, two timed chunks beside each other, and
back to 0 at a stop."""

import time
import unittest

import numpy as np

from modules.recording import ChunkStream, Player, PlayerSettings
from modules.recording.player.recording_folder import Timeline
from modules.recording.player.tests.test_frame_hand_over import FakeReaders


class ReadoutsTest(unittest.TestCase):
    def setUp(self) -> None:
        settings = PlayerSettings()
        settings.fps = 30.0
        self.player: Player = Player(settings)
        self.player.timeline = Timeline.of([30, 30], [0.0, 1.0])

    def show(self, chunk: int, frame_id: int) -> None:
        """Camera 0's frame of a chunk handed over, and the readouts updated at once."""
        for stream in (ChunkStream.VIDEO, ChunkStream.DEPTH):
            self.player.hand_over.submit_frame(0, stream, np.zeros(2), chunk, frame_id)
        self.player._gui_update_time = 0.0
        self.player._update_readouts()

    def test_the_playhead_stays_in_the_ending_chunk_while_the_next_is_followed(self) -> None:
        self.player.hand_over.start(FakeReaders(0, {0: [10.0 + i / 30.0 for i in range(30)]}))
        self.player.hand_over.follow(FakeReaders(1, {0: [11.0 + i / 30.0 for i in range(30)]}))
        self.player._set_play_chunk(1)          # as `_follow` sets it, while chunk 0's frames still go out
        self.show(0, 25)
        self.assertEqual(self.player.settings.current_chunk, 0)
        self.assertAlmostEqual(self.player.settings.playhead[0], 25 / 30.0)
        self.show(1, 2)
        self.assertEqual(self.player.settings.current_chunk, 1)
        self.assertAlmostEqual(self.player.settings.playhead[0], 32 / 30.0)

    def test_a_stop_puts_the_readouts_back_to_0(self) -> None:
        self.player.settings.playhead = [1.5, 2.0]
        self.player.settings.current_chunk = 1
        self.player.start()
        try:
            self.player.settings.stop = True
            deadline: float = time.perf_counter() + 2.0
            while self.player.settings.playhead[0] != 0.0 and time.perf_counter() < deadline:
                time.sleep(0.01)
        finally:
            self.player.stop()
        self.assertEqual(self.player.settings.playhead, [0.0, 2.0])
        self.assertEqual(self.player.settings.current_chunk, 0)

    def test_before_a_frame_is_shown_the_readouts_wait(self) -> None:
        self.player.hand_over.start(FakeReaders(0, {0: [10.0]}))
        self.player.settings.playhead = [7.0, 2.0]
        self.player._update_readouts()
        self.assertEqual(self.player.settings.playhead, [7.0, 2.0])


if __name__ == '__main__':
    unittest.main()
