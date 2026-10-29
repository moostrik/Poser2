"""Tests for the generated test workflow of the ComfyUI page (no ComfyUI server needed)."""

import os
import unittest

from apps.smile.comfy.testflow import CAPTURE, PROMPT, VIDEO, VIDEO_FILE, make_test_workflow
from apps.smile.comfy.workflow import Workflow
from apps.smile.settings import Settings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # apps/smile
WORKFLOW = os.path.join(ROOT, 'data', 'comfy', 'smile_workflow.json')


def sources(node: dict) -> set[str]:
    return {value[0] for value in node['inputs'].values() if isinstance(value, list) and len(value) == 2}


class TestTestflow(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.settings.generator.portrait.fixed_seed = True       # the app's build is then repeatable
        self.settings.sequence.grin_fps = 13.0
        self.wf = Workflow(WORKFLOW)
        p = self.settings.generator
        self.app = self.wf.build(CAPTURE, p.style_b, p)
        self.test = make_test_workflow(WORKFLOW, self.settings, 'b')

    def test_all_links_point_to_existing_nodes(self):
        for node_id, node in self.test.items():
            for source in sources(node):
                self.assertIn(source, self.test, f'{node_id} -> missing {source}')

    def test_what_the_app_sends_is_in_it(self):
        for node_id, node in self.app.items():
            other = self.test[node_id]
            self.assertEqual(other['class_type'], node['class_type'], node_id)
            for name, value in node['inputs'].items():
                self.assertEqual(other['inputs'][name], value, f'{node_id}.{name}')
        self.assertEqual(self.test[self.wf.load_image]['inputs']['image'], CAPTURE)

    def test_video_recorder_is_comfyui_s_own(self):
        # no node pack needed for the page: the grin becomes a video with ComfyUI's core video nodes
        self.assertEqual(self.test[VIDEO]['class_type'], 'CreateVideo')
        self.assertEqual(self.test[VIDEO]['inputs'], {'images': [self.wf.grin, 0], 'fps': 13.0})
        self.assertEqual(self.test[VIDEO_FILE]['class_type'], 'SaveVideo')
        self.assertEqual(self.test[VIDEO_FILE]['inputs']['video'], [VIDEO, 0])
        self.assertNotIn('VHS_VideoCombine', {node['class_type'] for node in self.test.values()})
        self.assertNotIn('LivePortraitCropper', {node['class_type'] for node in self.test.values()})

    def test_extras(self):
        overlay = next(node for node in self.test.values() if node['class_type'] == 'ImageBlend')
        self.assertEqual(overlay['inputs']['image2'], self.test[self.wf.portrait]['inputs']['images'])
        self.assertEqual(self.test[PROMPT]['inputs']['source'], [self.wf.suffix, 0])
        self.assertEqual(self.test[self.wf.style]['inputs']['text'], self.settings.generator.style_b.prompt)


if __name__ == '__main__':
    unittest.main()
