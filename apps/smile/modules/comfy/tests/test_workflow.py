"""Tests for adapting the exported ComfyUI workflow (no ComfyUI server needed)."""

import os
import unittest

from modules.comfy.settings import ComfySettings, Ease
from modules.comfy.styles import Styles
from modules.comfy.workflow import FIXED_SEED, MAX_SEED, Workflow, grin_box

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORKFLOW = os.path.join(ROOT, 'data', 'comfy', 'AIIAv3 portrait grin v3.json')


def links(workflow: dict) -> list[tuple[str, str, str]]:
    """(node, input, source node) for every link in the workflow."""
    return [(node_id, name, value[0]) for node_id, node in workflow.items()
            for name, value in node['inputs'].items() if isinstance(value, list) and len(value) == 2]


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow(WORKFLOW)
        self.settings = ComfySettings()
        self.settings.grin.frames = 120
        self.built = self.wf.build('smile_capture.png', 'a style', 'a negative', self.settings)

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
        self.assertEqual(self.built[self.wf.negative]['inputs']['text'], 'a negative')
        self.assertEqual(self.built[self.wf.grin]['inputs']['frames'], 120)
        self.assertEqual(self.built[self.wf.last_frame]['inputs']['batch_index'], 119)

    def test_settings_fill_the_nodes(self):
        s = ComfySettings()
        s.describe.instruction = 'describe'
        s.describe.max_tokens = 200
        s.describe.prefix = 'prefix'
        s.describe.suffix = ' suffix'
        s.portrait.steps = 8
        s.portrait.cfg = 2.0
        s.portrait.sampler = 'euler'
        s.portrait.scheduler = 'karras'
        s.portrait.embedding_strength = 0.8
        s.portrait.keypoint_strength = 0.7
        s.portrait.depth_strength = 0.6
        s.portrait.depth_end = 0.4
        s.lora.clothing = 0.9
        s.lora.looking = 0.0
        s.lora.dark = -0.3
        s.grin.ease = Ease.LINEAR
        s.grin.smile = 3.0
        s.grin.yaw_max = 10.0
        s.grin.stitching = False
        b = self.wf.build('x.png', 'style', 'negative', s)
        describe = b[self.wf.describe]['inputs']
        self.assertEqual((describe['custom_prompt'], describe['max_tokens']), ('describe', 200))
        # prefix + space + description, then + suffix
        prefix = b[self.wf.prefix]['inputs']
        self.assertEqual((prefix['string_a'], prefix['delimiter'], prefix['string_b']), ('prefix', ' ', [self.wf.describe, 0]))
        self.assertEqual(self.built[self.wf.prefix]['inputs']['delimiter'], '')     # no prefix: no leading space
        suffix = b[self.wf.suffix]['inputs']
        self.assertEqual((suffix['string_a'], suffix['string_b']), ([self.wf.prefix, 0], ' suffix'))
        self.assertEqual(b[self.wf.negative]['inputs']['text'], 'negative')
        sampler = b[self.wf.sampler]['inputs']
        self.assertEqual((sampler['steps'], sampler['cfg'], sampler['sampler_name'], sampler['scheduler']), (8, 2.0, 'euler', 'karras'))
        instantid = b[self.wf.instantid]['inputs']
        self.assertEqual((instantid['ip_weight'], instantid['cn_strength']), (0.8, 0.7))
        depth = b[self.wf.depth]['inputs']
        self.assertEqual((depth['strength'], depth['end_percent']), (0.6, 0.4))
        for name, strength in (('clothing', 0.9), ('looking', 0.0), ('dark', -0.3)):
            lora = b[self.wf.loras[name]]['inputs']
            self.assertEqual((lora['strength_model'], lora['strength_clip']), (strength, strength))
        grin = b[self.wf.grin]['inputs']
        self.assertEqual((grin['ease'], grin['smile'], grin['rotate_yaw_max'], grin['stitching']), ('linear', 3.0, 10.0, False))

    def test_seeds_random_and_in_range(self):
        other = self.wf.build('smile_capture.png', 'a style', 'a negative', self.settings)
        for node_id in (self.wf.sampler, self.wf.describe, self.wf.grin):
            seed = self.built[node_id]['inputs']['seed']
            self.assertTrue(1 <= seed <= MAX_SEED)
            self.assertNotEqual(seed, other[node_id]['inputs']['seed'])

    def test_fixed_seed(self):
        self.settings.portrait.fixed_seed = True
        first = self.wf.build('smile_capture.png', 'a style', 'a negative', self.settings)
        second = self.wf.build('smile_capture.png', 'a style', 'a negative', self.settings)
        for node_id in (self.wf.sampler, self.wf.describe, self.wf.grin):
            self.assertEqual(first[node_id]['inputs']['seed'], FIXED_SEED)
            self.assertEqual(second[node_id]['inputs']['seed'], FIXED_SEED)

    def test_full_preview_sampler(self):
        sampler = self.built[self.wf.sampler]
        self.assertEqual(sampler['class_type'], 'KSamplerFullPreview')
        decode = next(node for node in self.built.values() if node['class_type'] == 'VAEDecode')
        self.assertEqual(sampler['inputs']['vae'], decode['inputs']['vae'])
        self.settings.portrait.full_preview = False
        plain = self.wf.build('x.png', 'style', 'negative', self.settings)[self.wf.sampler]
        self.assertEqual(plain['class_type'], 'KSampler')
        self.assertNotIn('vae', plain['inputs'])

    def test_lora_chain(self):
        # checkpoint -> the LoRAs in a row -> InstantID (model) and the three text encoders (clip)
        checkpoint = next(node_id for node_id, node in self.built.items() if node['class_type'] == 'CheckpointLoaderSimple')
        source = checkpoint
        for node_id in self.wf.loras.values():
            inputs = self.built[node_id]['inputs']
            self.assertEqual((inputs['model'], inputs['clip']), ([source, 0], [source, 1]))
            source = node_id
        self.assertEqual(self.built[self.wf.instantid]['inputs']['model'], [source, 0])
        encoders = [node for node in self.built.values() if node['class_type'] == 'CLIPTextEncode']
        self.assertEqual(len(encoders), 3)
        for encoder in encoders:
            self.assertEqual(encoder['inputs']['clip'], [source, 1])

    def test_depth_outputs(self):
        for node_id in (self.wf.capture_depth, self.wf.last_depth):
            self.assertEqual(self.built[node_id]['class_type'], 'PreviewImage')

    def test_template_not_changed_by_build(self):
        again = self.wf.build('other.png', 'other', 'other negative', self.settings)
        self.assertEqual(self.built[self.wf.load_image]['inputs']['image'], 'smile_capture.png')
        self.assertEqual(again[self.wf.load_image]['inputs']['image'], 'other.png')

    def test_grin_crop_centred(self):
        self.assertEqual(self.wf.portrait_size, (896, 1120))
        self.assertEqual(grin_box((896, 1120), 0.857), (64, 80, 768, 960))
        crop = self.built[self.wf.grin_crop]['inputs']
        self.assertEqual((crop['x'], crop['y'], crop['width'], crop['height']), (64, 80, 768, 960))
        scale = self.built[self.wf.grin_scale]['inputs']
        self.assertEqual((scale['width'], scale['height']), (512, 640))
        for size in (0.5, 0.7, 0.93, 1.0):
            x, y, w, h = grin_box((896, 1120), size)
            self.assertEqual(w * 5, h * 4)                       # 4:5
            self.assertEqual((2 * x + w, 2 * y + h), (896, 1120))  # centred


class TestStyles(unittest.TestCase):
    def test_cycle(self):
        settings = ComfySettings()
        settings.style.repeat = 2
        settings.style.negative_a = 'negative a'
        settings.style.negative_b = 'negative b'
        styles = Styles(settings.style)
        ids = [styles.advance() for _ in range(3)]
        self.assertEqual(ids, ['0000', '0000', '0001'])
        self.assertEqual(settings.style.current, 'B')
        self.assertEqual((styles.current_text(), styles.current_negative()), (settings.style.style_b, 'negative b'))
        ids = [styles.advance() for _ in range(2)]
        self.assertEqual(ids, ['0001', '0000'])
        self.assertEqual(settings.style.current, 'A')
        self.assertEqual((styles.current_text(), styles.current_negative()), (settings.style.style_a, 'negative a'))


if __name__ == '__main__':
    unittest.main()
