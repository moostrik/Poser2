"""`Timeline`: the markers and places on a recording's timeline. `RecordingFolder`: what is a recording."""

import tempfile
import unittest
from pathlib import Path

from modules.camera import RecordingFolder, Timeline


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
