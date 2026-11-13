"""An `MP4` recording through the real encoders, where the iGPU is there: the color clip any player plays, at a
constant rate, with its metadata beside it, and no depth."""

import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from modules import video
from modules.camera import CameraFrame, FrameType
from modules.recording import Container, read_metadata
from modules.recording.tests.recordings import FPS, cam_frame, cam_tracklets, record

COUNT: int = 30


def feed(rec) -> None:
    for i in range(COUNT):
        t: float = 100.0 + i / FPS
        rec.submit_frame(cam_frame(0, i * 8 % 250, t))
        rec.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((120, 160), 1000, np.uint16), t))
        rec.submit_tracklets(cam_tracklets(0, t))
        time.sleep(0.003)


def probe(path: Path, entry: str) -> str:
    return subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', f'stream={entry}', '-of', 'csv=p=0',
                           str(path)], capture_output=True, text=True).stdout.strip()


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class Mp4Test(unittest.TestCase):
    def test_an_mp4_chunk_is_the_color_clip_with_its_metadata_beside(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            with self.assertLogs('modules.recording.recorder.recorder', 'WARNING'):     # depth asked for: color only
                recording = record(Path(folder), [FrameType.VIDEO, FrameType.DEPTH], 1, feed, container=Container.MP4)
            self.assertEqual(sorted(p.name for p in recording.iterdir()), ['0_C_000.mp4', '0_M_000.jsonl'])
            clip = recording / '0_C_000.mp4'
            self.assertEqual(probe(clip, 'codec_name'), 'h264')
            self.assertEqual(probe(clip, 'r_frame_rate'), '60/1')
            self.assertEqual(probe(clip, 'avg_frame_rate'), '60/1')         # constant: any player plays it at fps
            self.assertEqual(probe(clip, 'nb_frames'), str(COUNT))
            metadata = read_metadata(clip)
            assert metadata.info is not None
            self.assertFalse(metadata.info.depth)
            self.assertEqual(len(metadata.frames), COUNT)
            self.assertAlmostEqual(metadata.frames[5].timestamp, 100.0 + 5 / FPS)
            self.assertIsNotNone(metadata.frames[5].tracklets)


if __name__ == '__main__':
    unittest.main()
