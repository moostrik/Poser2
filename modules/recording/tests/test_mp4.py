"""An `MP4` recording through the real encoders, where the iGPU is there: the color clip any player plays, at a
constant rate, with its metadata beside it, and no depth; played back at its recorded times."""

import subprocess
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from modules import video
from modules.camera import CameraFrame, FrameType, RigCameraReadings
from modules.recording import Container, read_metadata
from modules.recording.replay import Camera
from modules.recording.tests.recordings import FPS, Played, cam_frame, cam_tracklets, collect, player, record

COUNT: int = 30


def feed(rec) -> None:
    for i in range(COUNT):
        t: float = 100.0 + i / FPS
        rec.submit_frame(cam_frame(0, i * 8 % 250, t))
        rec.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((120, 160), 1000, np.uint16), t))
        rec.submit_tracklets(cam_tracklets(0, t))
        time.sleep(0.003)


def probe(path: Path, entry: str) -> str:
    """One entry of the clip's first video stream, as ffprobe writes it."""
    args = ['-select_streams', 'v:0', '-show_entries', f'stream={entry}', '-of', 'csv=p=0', str(path)]
    out, _ = video.popen(args, subprocess.DEVNULL, subprocess.PIPE, command=video.FFPROBE).communicate(timeout=20)
    return out.decode().strip()


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
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

    def test_an_mp4_chunk_plays_at_its_recorded_times_with_its_metadata(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            recording = record(root, [FrameType.VIDEO], 1, feed, container=Container.MP4)
            play = player(root, 1)
            play.settings.folder = recording.name
            played: list[Played] = []
            collect(play, played)
            play.start()
            try:
                deadline = time.perf_counter() + 10.0
                while len(played) < COUNT and time.perf_counter() < deadline:
                    time.sleep(0.01)
            finally:
                play.settings.stop = True
                play.stop()
        first = played[:COUNT]
        self.assertEqual(len(first), COUNT)
        self.assertTrue(all(p[1] == FrameType.VIDEO for p in first))
        self.assertEqual([p[3] for p in first], [100.0 + i / FPS for i in range(COUNT)])
        assert first[3][5] is not None
        self.assertIsNotNone(first[3][5].tracklets)
        captures = np.array([p[4] for p in first[5:]], dtype=np.float64)       # on the play's clock, 1/60 s apart
        np.testing.assert_allclose(np.diff(captures), 1.0 / FPS, atol=1e-6)

    def test_the_turn_and_the_flips_come_back_as_flags(self) -> None:
        """Turned and flipped frames, as smile's camera delivers them, are stored as delivered and play out of the
        replay camera with the same flags."""
        def turned(rec) -> None:
            for i in range(COUNT):
                rec.submit_frame(replace(cam_frame(0, 100, 100.0 + i / FPS), rotate90=True, flip_v=True, flip_h=True))
                time.sleep(0.003)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            recording = record(root, [FrameType.VIDEO], 1, turned, container=Container.MP4)
            metadata = read_metadata(recording / '0_C_000.mp4')
            assert metadata.info is not None
            self.assertEqual((metadata.info.width, metadata.info.height), (320, 240))     # as delivered
            self.assertTrue(metadata.info.rotate90 and metadata.info.flip_v and metadata.info.flip_h)

            play = player(root, 1)
            play.settings.folder = recording.name
            readings = RigCameraReadings()
            readings.initialize()
            camera = Camera(play, readings, None, None)
            frames: list[CameraFrame] = []
            camera.add_frame_callback(frames.append)
            camera.start()
            play.start()
            try:
                deadline = time.perf_counter() + 10.0
                while len(frames) < 5 and time.perf_counter() < deadline:
                    time.sleep(0.01)
            finally:
                play.settings.stop = True
                play.stop()
                camera.stop()
                camera.join()
        self.assertGreaterEqual(len(frames), 5)
        for frame in frames:
            self.assertEqual(frame.image.shape[:2], (240, 320))
            self.assertTrue(frame.rotate90 and frame.flip_v and frame.flip_h)


if __name__ == '__main__':
    unittest.main()
