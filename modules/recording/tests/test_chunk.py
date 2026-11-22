"""A chunk's metadata: its lines written and read back; its parts merged, and finished after a crash; a recording's
folder name."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules import video
from modules.camera import CameraTracklet, DepthAlignment, FrameReadings, TrackingStatus
from modules.recording.chunk import ChunkInfo, ChunkStream, FrameMetadata, finish_unmerged, frame_line, info_line, make_chunk_name, \
    make_file_name, make_folder_name, make_metadata_name, merge_chunk, parse_metadata, read_metadata
from modules.video import StreamWriter
from modules.recording.tests.recordings import LENS
from modules.utils import Rect


class MetadataTest(unittest.TestCase):
    def test_the_lines_read_back(self) -> None:
        info = ChunkInfo(900, 720, LENS, True, 60.0)
        points = np.array([[0.1, 0.2, 0.9]] * 17, dtype=np.float32)
        tracklet = CameraTracklet(cam_id=2, id=4, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=7, timestamp=1.5, keypoints=points)
        box = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        text = (info_line(info) + frame_line(0, FrameMetadata(1.5, FrameReadings(-30.0, 1.0, 1000, 100), 0.01, [tracklet], 0.033,
                                                              {3: (box, points)}, 0.05))
                + frame_line(1, FrameMetadata(1.6, FrameReadings(math.nan, 1.0, 1000, 100), None, None, None))
                + frame_line(2, FrameMetadata(1.7, None, 0.01, [], 0.03)))
        metadata = parse_metadata(text, cam_id=2)
        self.assertEqual(metadata.info, info)
        first = metadata.frames[0]
        self.assertEqual((first.timestamp, first.readings, first.video_latency, first.tracklet_latency),
                         (1.5, FrameReadings(-30.0, 1.0, 1000, 100), 0.01, 0.033))
        assert first.tracklets is not None and first.tracklets[0].keypoints is not None
        self.assertEqual((first.tracklets[0].cam_id, first.tracklets[0].id, first.tracklets[0].status, first.tracklets[0].roi, first.tracklets[0].age),
                         (2, 4, TrackingStatus.TRACKED, Rect(0.1, 0.2, 0.3, 0.4), 7))
        np.testing.assert_array_almost_equal(first.tracklets[0].keypoints, points)
        assert first.poses is not None
        np.testing.assert_array_almost_equal(first.poses[3][0], box)
        np.testing.assert_array_almost_equal(first.poses[3][1], points)
        self.assertEqual(first.pose_latency, 0.05)
        self.assertIsNone(metadata.frames[1].poses)
        assert metadata.frames[1].readings is not None
        self.assertTrue(math.isnan(metadata.frames[1].readings.tilt))
        self.assertIsNone(metadata.frames[1].tracklets)
        self.assertIsNone(metadata.frames[2].readings)
        self.assertEqual(metadata.frames[2].tracklets, [])

    def test_the_depth_alignment_reads_back(self) -> None:
        alignment = DepthAlignment([[212.4, 0.0, 240.6], [0.0, 212.4, 149.4], [0.0, 0.0, 1.0]], LENS, (896, 720),
                                   [[1.0, 0.0, 0.0, -0.0376], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        info = ChunkInfo(896, 720, LENS, True, 60.0, alignment)
        self.assertEqual(parse_metadata(info_line(info)).info, info)

    def test_a_line_cut_off_by_a_crash_ends_the_list(self) -> None:
        text = info_line(ChunkInfo(900, 720, LENS, True, 60.0)) + frame_line(0, FrameMetadata(1.5)) + '{"frame": 1, "ti'
        self.assertEqual(len(parse_metadata(text).frames), 1)


PARTS: int = 10         # frames in each part of a merged chunk


def write_parts(folder: Path, confidence: bool = True) -> tuple[Path, Path, Path | None, Path]:
    """A chunk's parts as the recorder leaves them in its temp folder: color, depth, the confidence when asked, and
    the metadata, `PARTS` frames each."""
    parts: dict[ChunkStream, Path] = {}
    writers = {ChunkStream.VIDEO: StreamWriter(video.ClipCodec.H264, video.Hardware.iGPU, 1),
               ChunkStream.DEPTH: StreamWriter(video.DEPTH_CODEC, video.DEPTH_HARDWARE, 1),
               ChunkStream.CONFIDENCE: StreamWriter(video.DEPTH_CODEC, video.DEPTH_HARDWARE, 1, b_frames=False)}
    if not confidence:
        del writers[ChunkStream.CONFIDENCE]
    for stream, writer in writers.items():
        parts[stream] = folder / make_file_name(0, stream, 0, '.mkv')
        writer.start(str(parts[stream]), 60.0)
    for i in range(PARTS):
        writers[ChunkStream.VIDEO].submit_frame(np.full((240, 320, 3), i * 20, np.uint8))
        writers[ChunkStream.DEPTH].submit_frame(np.full((300, 480), 1000 + i * 30, np.uint16))
        if confidence:
            writers[ChunkStream.CONFIDENCE].submit_frame(np.full((300, 480), 100 + i * 10, np.uint8))
    for writer in writers.values():
        writer.stop()
    metadata = folder / make_metadata_name(0, 0)
    metadata.write_text(info_line(ChunkInfo(320, 240, LENS, True, 60.0, None, True, True))
                        + ''.join(frame_line(i, FrameMetadata(i / 60.0)) for i in range(PARTS)), encoding='utf-8')
    return parts[ChunkStream.VIDEO], parts[ChunkStream.DEPTH], parts.get(ChunkStream.CONFIDENCE), metadata


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
class MergeTest(unittest.TestCase):
    def test_three_parts_merge_into_one_chunk_in_their_order(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            color, depth, confidence, metadata = write_parts(root)
            output = root / make_chunk_name(0, 0)
            self.assertTrue(merge_chunk(color, depth, confidence, metadata, output))
            for stream in ChunkStream:
                self.assertEqual(video.probe(str(output), int(stream)).frames, PARTS)
            self.assertEqual((video.probe(str(output), ChunkStream.CONFIDENCE).width, video.probe(str(output), ChunkStream.DEPTH).width), (480, 480))
            info = read_metadata(output).info
            assert info is not None
            self.assertTrue(info.confidence)
            self.assertEqual(sorted(p.name for p in root.iterdir()), [output.name])     # the parts gone

    def test_what_a_crash_left_is_finished_with_its_three_parts(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            (root / 'temp' / 'take').mkdir(parents=True)
            write_parts(root / 'temp' / 'take')
            finish_unmerged(root / 'temp', root / 'recordings')
            chunk = root / 'recordings' / 'take' / make_chunk_name(0, 0)
            self.assertTrue(chunk.is_file())
            self.assertEqual(video.probe(str(chunk), ChunkStream.CONFIDENCE).frames, PARTS)

    def test_a_chunk_whose_confidence_is_missing_is_left_as_it_is(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            (root / 'temp' / 'take').mkdir(parents=True)
            write_parts(root / 'temp' / 'take', confidence=False)        # its info line says it has one
            finish_unmerged(root / 'temp', root / 'recordings')
            self.assertFalse((root / 'recordings' / 'take' / make_chunk_name(0, 0)).exists())
            self.assertEqual(len(list((root / 'temp' / 'take').iterdir())), 3)


class FolderNameTest(unittest.TestCase):
    def test_two_recordings_started_in_one_second_get_two_folders(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            with patch('time.strftime', return_value='20261113-120000'):
                names: list[str] = []
                for _ in range(3):
                    names.append(make_folder_name('take', Path(parent)))
                    (Path(parent) / names[-1]).mkdir()
        self.assertEqual(names, ['20261113-120000_take', '20261113-120000_take-2', '20261113-120000_take-3'])


if __name__ == '__main__':
    unittest.main()
