"""`ChunkReaders`' search of a camera's frames by recorded time."""

import unittest

from modules.camera import FrameMetadata
from modules.camera.recording.player.chunk_readers import first_at


def frames(times: list[float], latency: float = 0.03) -> list[FrameMetadata]:
    return [FrameMetadata(t, video_latency=latency) for t in times]


class FirstAtTest(unittest.TestCase):
    def test_the_first_frame_at_or_after_a_time(self) -> None:
        cam = frames([10.0, 10.02, 10.06, 10.08])
        self.assertEqual(first_at(cam, 10.02), 1)
        self.assertEqual(first_at(cam, 10.03), 2)       # 10.04 dropped: the next there is
        self.assertEqual(first_at(cam, 9.0), 0)
        self.assertEqual(first_at(cam, 11.0), 4)


if __name__ == '__main__':
    unittest.main()
