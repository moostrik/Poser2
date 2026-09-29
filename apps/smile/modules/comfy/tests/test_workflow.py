"""Tests for adapting the exported ComfyUI workflow (no ComfyUI server needed)."""

import os
import unittest

from modules.comfy.styles import Styles
from modules.comfy.workflow import MAX_SEED, Workflow

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORKFLOW = os.path.join(ROOT, 'data', 'comfy', 'AIIAv3 portrait grin v2.json')
STYLES = os.path.join(ROOT, 'prompts', 'smile.json')


def links(workflow: dict) -> list[tuple[str, str, str]]:
    """(node, input, source node) for every link in the workflow."""
    return [(node_id, name, value[0]) for node_id, node in workflow.items()
            for name, value in node['inputs'].items() if isinstance(value, list) and len(value) == 2]


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow(WORKFLOW)
        self.built = self.wf.build('smile_capture.png', 'a style', 120)

    def test_all_links_point_to_existing_nodes(self):
        for node_id, name, source in links(self.built):
            self.assertIn(source, self.built, f'{node_id}.{name} -> missing {source}')

    def test_face_crop_removed_and_rewired(self):
        classes = {node['class_type'] for node in self.built.values()}
        self.assertNotIn('LivePortraitCropper', classes)
        self.assertNotIn('LivePortraitLoadFaceAlignmentCropper', classes)
        # the old crop output (the pass-through preview) now reads the uploaded image
        sources = {source for _, _, source in links(self.built)}
        self.assertIn(self.wf.load_image, sources)

    def test_unused_outputs_removed(self):
        classes = {node['class_type'] for node in self.built.values()}
        self.assertNotIn('VHS_VideoCombine', classes)
        self.assertNotIn('ImageBlend', classes)
        self.assertNotIn('SaveImage', classes)      # the portrait is a preview, nothing piles up in output/
        self.assertEqual(self.built[self.wf.portrait]['class_type'], 'PreviewImage')

    def test_per_run_values(self):
        self.assertEqual(self.built[self.wf.load_image]['inputs']['image'], 'smile_capture.png')
        self.assertEqual(self.built[self.wf.style]['inputs']['text'], 'a style')
        self.assertEqual(self.built[self.wf.grin]['inputs']['frames'], 120)
        self.assertEqual(self.built[self.wf.last_frame]['inputs']['batch_index'], 119)

    def test_seeds_random_and_in_range(self):
        other = self.wf.build('smile_capture.png', 'a style', 120)
        for node_id in (self.wf.sampler, self.wf.describe, self.wf.grin):
            seed = self.built[node_id]['inputs']['seed']
            self.assertTrue(1 <= seed <= MAX_SEED)
            self.assertNotEqual(seed, other[node_id]['inputs']['seed'])

    def test_full_preview_sampler(self):
        sampler = self.built[self.wf.sampler]
        self.assertEqual(sampler['class_type'], 'KSamplerFullPreview')
        decode = next(node for node in self.built.values() if node['class_type'] == 'VAEDecode')
        self.assertEqual(sampler['inputs']['vae'], decode['inputs']['vae'])

    def test_depth_outputs(self):
        for node_id in (self.wf.capture_depth, self.wf.last_depth):
            self.assertEqual(self.built[node_id]['class_type'], 'PreviewImage')

    def test_template_not_changed_by_build(self):
        again = self.wf.build('other.png', 'other', 90)
        self.assertEqual(self.built[self.wf.load_image]['inputs']['image'], 'smile_capture.png')
        self.assertEqual(again[self.wf.load_image]['inputs']['image'], 'other.png')

    def test_grin_crop(self):
        self.assertEqual(self.wf.portrait_size, (896, 1120))
        x, y, w, h = self.wf.grin_crop
        self.assertAlmostEqual(w, 768 / 896)
        self.assertAlmostEqual(h, 960 / 1120)


class TestStyles(unittest.TestCase):
    def test_cycle(self):
        styles = Styles(STYLES)
        ids = [styles.advance(repeat=2) for _ in range(5)]
        self.assertEqual(ids, ['0000', '0000', '0001', '0001', '0000'])
        self.assertTrue(styles.current_text())


if __name__ == '__main__':
    unittest.main()
