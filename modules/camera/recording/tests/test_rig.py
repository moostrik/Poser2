"""Several cameras recorded and played back through the real encoders: two with unlike data, and a
rig of three unsynced cameras with the same data, color only and with depth."""

import tempfile
import time
import unittest
from pathlib import Path
from typing import Callable

import numpy as np

from modules import video
from modules.camera import CameraFrame, FramePoses, FrameType, Recorder, RecorderSettings, RecordingFolder, \
    make_chunk_name, read_metadata
from modules.camera.recording.player.chunk_readers import ChunkReaders, Decoding
from modules.camera.recording.tests.recordings import POINTS, poses, stream_frames, cam_frame, cam_tracklets, record, \
    player, Played, collect, FPS, SPLIT, COUNT


def two_cameras(rec: Recorder) -> None:
    """Camera 0 with color, depth, tracklets and poses; camera 1 color only, 3 ms after camera 0, its every fifth
    frame lost live. A split after frame `SPLIT - 1`."""
    for i in range(COUNT):
        if i == SPLIT:
            time.sleep(0.3)
            RecorderSettings.split.fire(rec.settings)
            time.sleep(0.3)
        t: float = 100.0 + i / FPS
        rec.submit_frame(cam_frame(0, i * 4 % 250, t))
        rec.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((120, 160), 1000 + i * 30, np.uint16), t))
        rec.submit_tracklets(cam_tracklets(0, t))
        rec.expect_poses(0, t, True)
        rec.submit_poses(poses(t))
        if i % 5 != 4:
            rec.submit_frame(cam_frame(1, i * 4 % 250, t + 0.003))
        time.sleep(0.003)


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class SeveralCamerasTest(unittest.TestCase):
    """Two cameras with unlike data recorded and played back through the real encoders."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.folder.name)
        cls.recording = record(cls.root, [FrameType.VIDEO, FrameType.DEPTH], 2, two_cameras)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def test_the_chunks_are_cut_at_one_capture_time(self) -> None:
        names = sorted(p.name for p in self.recording.iterdir() if p.suffix == '.mkv')
        self.assertEqual(names, [make_chunk_name(c, k) for c in (0, 1) for k in (0, 1)])
        lost = lambda lo, hi: sum(1 for i in range(lo, hi) if i % 5 == 4)
        for cam, chunk, count, depth in ((0, 0, SPLIT, True), (0, 1, COUNT - SPLIT, True),
                                         (1, 0, SPLIT - lost(0, SPLIT), False), (1, 1, COUNT - SPLIT - lost(SPLIT, COUNT), False)):
            path = self.recording / make_chunk_name(cam, chunk)
            metadata = read_metadata(path, cam)
            assert metadata.info is not None
            self.assertEqual(metadata.info.depth, depth)
            self.assertEqual(len(metadata.frames), count, (cam, chunk))
            self.assertEqual(stream_frames(path, 0), count)
            if depth:
                self.assertEqual(stream_frames(path, 1), count)
        cut: float = 100.0 + (SPLIT - 0.5) / FPS
        for cam in (0, 1):
            self.assertLess(read_metadata(self.recording / make_chunk_name(cam, 0), cam).frames[-1].timestamp, cut)
            self.assertGreater(read_metadata(self.recording / make_chunk_name(cam, 1), cam).frames[0].timestamp, cut)

    def test_both_play_on_the_recorded_time_with_their_own_streams(self) -> None:
        play = player(self.root, 2)
        play.settings.folder = self.recording.name
        played: list[Played] = []
        collect(play, played)
        play.start()
        try:
            deadline = time.perf_counter() + 10.0
            expected: int = 2 * COUNT + COUNT - (COUNT // 5)
            while len(played) < expected and time.perf_counter() < deadline:
                time.sleep(0.01)
        finally:
            play.settings.stop = True
            play.stop()

        def first_loop(cam: int, kind: FrameType) -> list[Played]:
            """A camera's frames of a kind up to where the loop starts again: its recorded time going back."""
            own = [p for p in played if p[0] == cam and p[1] == kind and p[3] is not None]
            return own[:next((i for i in range(1, len(own)) if (own[i][3] or 0.0) < (own[i - 1][3] or 0.0)), len(own))]

        colors = {cam: first_loop(cam, FrameType.VIDEO) for cam in (0, 1)}
        self.assertEqual(len(colors[0]), COUNT)
        self.assertEqual(len(colors[1]), COUNT - COUNT // 5)
        self.assertFalse([p for p in played if p[0] == 1 and p[1] == FrameType.DEPTH])
        self.assertEqual(len(first_loop(0, FrameType.DEPTH)), len(colors[0]))
        for p in colors[0][1:]:     # the first frame went out before a tracklet said the camera sends them
            assert p[5] is not None and p[5].tracklets is not None and p[5].poses is not None
        for p in colors[1]:
            assert p[5] is not None
            self.assertIsNone(p[5].tracklets)
            self.assertIsNone(p[5].poses)
        # One origin for both cameras across the split: each capture on the play's clock is its recorded one moved
        # by the same amount, a catch-up would move it; the first frames are left out, where the decoders start
        offsets = np.array([p[4] - p[3] for cam in (0, 1) for p in colors[cam]
                            if p[3] is not None and p[4] is not None and p[3] > 100.0 + 5.0 / FPS])
        self.assertLess(float(offsets.max() - offsets.min()), 0.002)
        for cam in (0, 1):
            recorded = [p[3] for p in colors[cam]]
            self.assertEqual(recorded, sorted(recorded))

    def test_a_seek_lands_both_cameras_at_one_recorded_time(self) -> None:
        folder = RecordingFolder.find(self.root / 'recordings')[self.recording.name]
        decoding = Decoding(False, False, *video.hwaccel(video.CoderType.iGPU), FPS, '')
        for lo in (3, 4, 9):
            readers = ChunkReaders(folder, 1, lo, None, 2, [FrameType.VIDEO, FrameType.DEPTH], lambda *_: None, decoding)
            try:
                reference = readers.metadata[0].frames[lo].timestamp
                first, _ = readers._range(1, lo, None)
                landed = readers.metadata[1].frames[first].timestamp
                self.assertGreaterEqual(landed, reference - 0.5 / FPS)     # its frame of the same round, or the next one where that was lost
                self.assertLess(landed - reference, 1.5 / FPS)
                self.assertEqual(readers.types, {0: {FrameType.VIDEO, FrameType.DEPTH}, 1: {FrameType.VIDEO}})
            finally:
                readers.stop()
                readers.join()


RIG: int = 3                        # cameras of one rig, as hd_trio's
OFFSETS: list[float] = [0.0, 0.007, 0.013]  # each camera's capture after camera 0's: the cameras run unsynced
LOST: list[int] = [7, 11, 13]       # each camera loses every so many frames live, unlike the others


def rig(with_depth: bool) -> Callable[[Recorder], None]:
    """Every camera of a rig with the same data: color, tracklets and poses, and depth for all or none. A split after
    frame `SPLIT - 1`."""
    def feed(rec: Recorder) -> None:
        for i in range(COUNT):
            if i == SPLIT:
                time.sleep(0.3)
                RecorderSettings.split.fire(rec.settings)
                time.sleep(0.3)
            for cam in range(RIG):
                if i % LOST[cam] == LOST[cam] - 1:
                    continue
                t: float = 100.0 + i / FPS + OFFSETS[cam]
                rec.submit_frame(cam_frame(cam, (i * 4 + cam * 60) % 250, t))
                if with_depth:
                    rec.submit_frame(CameraFrame(cam, FrameType.DEPTH, np.full((120, 160), 1000 + i * 30, np.uint16), t))
                rec.submit_tracklets(cam_tracklets(cam, t))
                rec.expect_poses(cam, t, True)
                rec.submit_poses(FramePoses(cam, t, {3: (np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), POINTS)}, t + 0.05))
            time.sleep(0.004)
    return feed


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class RigTest(unittest.TestCase):
    """The cameras of one rig, as hd_trio's and white_space's: one folder, every camera with the same data, recorded
    and played back together in step."""

    def check(self, with_depth: bool) -> None:
        kept = lambda cam, lo, hi: sum(1 for i in range(lo, hi) if i % LOST[cam] != LOST[cam] - 1)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            types = [FrameType.VIDEO, FrameType.DEPTH] if with_depth else [FrameType.VIDEO]
            recording = record(root, types, RIG, rig(with_depth))

            names = sorted(p.name for p in recording.iterdir() if p.suffix == '.mkv')
            self.assertEqual(names, sorted(make_chunk_name(c, k) for c in range(RIG) for k in (0, 1)))
            cut: float = 100.0 + (SPLIT - 0.5) / FPS
            for cam in range(RIG):
                for chunk, (lo, hi) in enumerate(((0, SPLIT), (SPLIT, COUNT))):
                    path = recording / make_chunk_name(cam, chunk)
                    metadata = read_metadata(path, cam)
                    assert metadata.info is not None
                    self.assertEqual(metadata.info.depth, with_depth)
                    self.assertEqual(len(metadata.frames), kept(cam, lo, hi), (cam, chunk))
                    self.assertEqual(stream_frames(path, 0), len(metadata.frames))
                    if with_depth:
                        self.assertEqual(stream_frames(path, 1), len(metadata.frames))
                    for frame in metadata.frames[1:]:       # the first went out before its tracklets said they come
                        self.assertIsNotNone(frame.tracklets)
                        self.assertIsNotNone(frame.poses)
                    # one cut for the rig: every camera's chunk 0 ends and chunk 1 starts at the same moment
                    if chunk == 0:
                        self.assertLess(metadata.frames[-1].timestamp, cut + OFFSETS[cam])
                    else:
                        self.assertGreater(metadata.frames[0].timestamp, cut)

            play = player(root, RIG)
            play.settings.folder = recording.name
            played: list[Played] = []
            drifts: list[float] = []
            collect(play, played)
            play.add_frame_callback(lambda cam, kind, image, frame, captured: drifts.append(play.get_drift()))
            play.start()
            try:
                deadline = time.perf_counter() + 10.0
                per_loop: int = sum(kept(c, 0, COUNT) for c in range(RIG)) * (2 if with_depth else 1)
                while len(played) < per_loop + 10 and time.perf_counter() < deadline:
                    time.sleep(0.01)
            finally:
                play.settings.stop = True
                play.stop()

        def first_loop(cam: int, kind: FrameType) -> list[Played]:
            own = [p for p in played if p[0] == cam and p[1] == kind and p[3] is not None]
            return own[:next((i for i in range(1, len(own)) if (own[i][3] or 0.0) < (own[i - 1][3] or 0.0)), len(own))]

        offsets: list[float] = []
        for cam in range(RIG):
            colors = first_loop(cam, FrameType.VIDEO)
            self.assertEqual(len(colors), kept(cam, 0, COUNT), cam)
            self.assertEqual(len(first_loop(cam, FrameType.DEPTH)), len(colors) if with_depth else 0)
            recorded = [(i * 4 + cam * 60) % 250 for i in range(COUNT) if i % LOST[cam] != LOST[cam] - 1]
            self.assertLessEqual(max(abs(p[2] - v) for p, v in zip(colors, recorded)), 2)    # its own frames, in order
            for p in colors[1:]:
                assert p[5] is not None and p[5].tracklets is not None and p[5].poses is not None
            offsets += [p[4] - p[3] for p in colors if p[3] is not None and p[4] is not None and p[3] > 100.0 + 5.0 / FPS]
        # one clock for the rig across the split: every camera's frames at their recorded times on one origin
        self.assertLess(max(offsets) - min(offsets), 0.002)
        # the cameras play side by side: never more than a frame and a camera's offset apart
        self.assertLess(float(np.median(drifts)), 1000.0 / FPS + 13.0 + 2.0)

    def test_a_rig_without_depth(self) -> None:
        self.check(False)

    def test_a_rig_with_depth(self) -> None:
        self.check(True)


if __name__ == '__main__':
    unittest.main()
