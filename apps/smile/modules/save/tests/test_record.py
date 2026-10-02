"""A record survives the trip into a JPEG and back, and the JPEG stays the same image."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from modules.comfy.settings import PipelineSettings
from modules.save import record


class TestRecord(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, 'portrait.jpg')
        self.image = np.random.default_rng(1).integers(0, 255, (64, 48, 3), dtype=np.uint8)
        ok, jpeg = cv2.imencode('.jpg', self.image)
        self.jpeg = jpeg.tobytes()

    def write(self, data: bytes) -> None:
        with open(self.path, 'wb') as f:
            f.write(data)

    def test_round_trip(self):
        text = json.dumps({'id': '2026_10_02_14_03_11', 'style': 'B', 'description': 'café, “quoted”', 'seed': 7})
        self.write(record.embed_jpeg(self.jpeg, text))
        self.assertEqual(record.read_jpeg(self.path), text)
        self.assertEqual(record.read(self.path)['style'], 'B')
        # the same image as without the record
        np.testing.assert_array_equal(cv2.imread(self.path), cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR))

    def test_a_long_text_spans_segments(self):
        text = json.dumps({'description': 'x' * 150000})
        self.write(record.embed_jpeg(self.jpeg, text))
        self.assertEqual(record.read_jpeg(self.path), text)
        self.assertIsNotNone(cv2.imread(self.path))

    def test_a_jpeg_without_a_record(self):
        self.write(self.jpeg)
        self.assertIsNone(record.read_jpeg(self.path))
        self.assertIsNone(record.read(self.path))

    def test_a_record_as_a_preset(self):
        made = PipelineSettings()
        made.style_b.prompt = 'the style'
        made.portrait.steps = 9
        made.grin.frames = 60
        made.grin.midpoint = 0.5
        visitor = {'style': 'B', 'style_settings': made.style_b.to_dict(), 'portrait': made.portrait.to_dict(),
                   'grin': {'frames': 60, 'ease': made.grin.to_dict()['ease'], 'midpoint': 0.5,
                            'peak': {'smile': 3.21, 'yaw': -7.5}, 'peak_at': {'smile': 0.6, 'yaw': 0.4}}}
        settings = PipelineSettings()
        style_a = settings.style_a.to_dict()
        settings.update_from_dict(json.loads(json.dumps(record.to_preset(visitor)))['pipeline'])
        self.assertEqual(settings.style_b.prompt, 'the style')
        self.assertEqual(settings.style_a.to_dict(), style_a)       # the other style is not touched
        self.assertEqual(settings.portrait.steps, 9)
        g = settings.grin
        self.assertEqual((g.frames, g.midpoint), (60, 0.5))
        # every range closed on the value that was drawn
        self.assertEqual((g.smile_min, g.smile_max, g.yaw_min, g.yaw_max), (3.21, 3.21, -7.5, -7.5))

    def test_a_record_of_the_first_shape_as_a_preset(self):
        self.assertEqual(record.to_preset({'style': 'A', 'settings': {'pipeline': {}}}), {'pipeline': {}})

    def test_a_comment_that_is_no_record(self):
        self.write(record.embed_jpeg(self.jpeg, 'made by a camera'))
        self.assertIsNone(record.read(self.path))


if __name__ == '__main__':
    unittest.main()
