"""Tests for the generated test workflow of the ComfyUI page (no ComfyUI server needed)."""

import os
import unittest

from modules.comfy.testflow import CAPTURE, OUTPUT_CLASSES, PROMPT, SWITCH, make_test_workflow
from modules.comfy.workflow import Workflow
from modules.smile_settings import Settings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORKFLOW = os.path.join(ROOT, 'data', 'comfy', 'smile_workflow.json')


def sources(node: dict) -> set[str]:
    return {value[0] for value in node['inputs'].values() if isinstance(value, list) and len(value) == 2}


class TestTestflow(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.settings.pipeline.portrait.fixed_seed = True       # the app's build is then repeatable
        self.settings.sequence.grin_fps = 13.0
        self.wf = Workflow(WORKFLOW)
        p = self.settings.pipeline
        self.app = self.wf.build(CAPTURE, p.style.style_b, p.style.negative_b, p)
        self.test = make_test_workflow(WORKFLOW, self.settings, 'b')

    def test_all_links_point_to_existing_nodes(self):
        for node_id, node in self.test.items():
            for source in sources(node):
                self.assertIn(source, self.test, f'{node_id} -> missing {source}')

    def test_what_the_app_sends_is_in_it(self):
        image = [self.wf.load_image, 0]
        for node_id, node in self.app.items():
            other = self.test[node_id]
            self.assertEqual(other['class_type'], node['class_type'], node_id)
            for name, value in node['inputs'].items():
                self.assertEqual(other['inputs'][name], [SWITCH, 0] if value == image else value, f'{node_id}.{name}')

    def test_input_switch(self):
        switch = self.test[SWITCH]['inputs']
        self.assertFalse(switch['switch'])
        self.assertEqual(switch['on_false'], [self.wf.load_image, 0])
        self.assertEqual(self.test[self.wf.load_image]['inputs']['image'], CAPTURE)
        # the on side is the face crop, back to the same photo
        chain, todo = set(), [switch['on_true'][0]]
        while todo:
            node_id = todo.pop()
            if node_id not in chain and node_id != self.wf.load_image:
                chain.add(node_id)
                todo.extend(sources(self.test[node_id]))
        self.assertIn('LivePortraitCropper', {self.test[node_id]['class_type'] for node_id in chain})
        self.assertIn(self.wf.load_image, {s for node_id in chain for s in sources(self.test[node_id])})
        # only the switch reads the crop, and no output node: it stays off while the switch is off
        for node_id, node in self.test.items():
            if node_id in chain:
                self.assertNotIn(node['class_type'], OUTPUT_CLASSES)
            elif node_id != SWITCH:
                self.assertFalse(sources(node) & chain, node_id)

    def test_extras(self):
        video = next(node for node in self.test.values() if node['class_type'] == 'VHS_VideoCombine')
        self.assertEqual(video['inputs']['images'], [self.wf.grin, 0])
        self.assertEqual(video['inputs']['frame_rate'], 13.0)
        overlay = next(node for node in self.test.values() if node['class_type'] == 'ImageBlend')
        self.assertEqual(overlay['inputs']['image2'], self.test[self.wf.portrait]['inputs']['images'])
        self.assertEqual(self.test[PROMPT]['inputs']['source'], [self.wf.suffix, 0])
        self.assertEqual(self.test[self.wf.style]['inputs']['text'], self.settings.pipeline.style.style_b)


if __name__ == '__main__':
    unittest.main()
