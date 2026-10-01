"""Tests for adapting the exported ComfyUI workflow (no ComfyUI server needed)."""

import os
import random
import unittest

from modules.comfy.settings import PipelineSettings, Ease, Prefer
from modules.comfy.styles import Styles
from modules.comfy.workflow import FIXED_SEED, GRIN_ROTATIONS, GRIN_TARGETS, MAX_SEED, Workflow

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORKFLOW = os.path.join(ROOT, 'data', 'comfy', 'smile_workflow.json')


def links(workflow: dict) -> list[tuple[str, str, str]]:
    """(node, input, source node) for every link in the workflow."""
    return [(node_id, name, value[0]) for node_id, node in workflow.items()
            for name, value in node['inputs'].items() if isinstance(value, list) and len(value) == 2]


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow(WORKFLOW)
        self.settings = PipelineSettings()
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
        s = PipelineSettings()
        s.describe.instruction = 'describe'
        s.describe.max_tokens = 200
        s.describe.repetition_penalty = 1.1
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
        s.grin.smile_min = s.grin.smile_max = 3.0
        s.grin.yaw_min = s.grin.yaw_max = 10.0
        s.grin.stitching = False
        b = self.wf.build('x.png', 'style', 'negative', s)
        describe = b[self.wf.describe]['inputs']
        self.assertEqual((describe['custom_prompt'], describe['max_tokens']), ('describe', 200))
        self.assertEqual((b[self.wf.describe]['class_type'], describe['repetition_penalty']), ('AILab_QwenVL_Advanced', 1.1))
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
        self.assertEqual((grin['ease'], grin['smile'], grin['stitching']), ('linear', 3.0, False))
        self.assertEqual((grin['rotate_yaw_min'], grin['rotate_yaw_max']), (10.0, 10.0))

    def test_peaks_random_within_the_range(self):
        g = self.settings.grin
        g.aaa_min, g.aaa_max = -10.0, -20.0      # max may be the lower number
        g.yaw_min, g.yaw_max = -15.0, 15.0
        peaks = [self.wf.build('x.png', 'style', 'negative', self.settings)[self.wf.grin]['inputs'] for _ in range(20)]
        for grin in peaks:
            self.assertTrue(-20.0 <= grin['aaa'] <= -10.0)
            self.assertTrue(-15.0 <= grin['rotate_yaw_min'] <= 15.0)
            self.assertEqual(grin['rotate_yaw_min'], grin['rotate_yaw_max'])     # the node gets the pick, not the range
            self.assertEqual(grin['smile'], 2.0)                                 # min == max: no variation
            self.assertFalse(grin['still'])
        self.assertGreater(len({grin['aaa'] for grin in peaks}), 1)
        self.settings.portrait.fixed_seed = True
        self.assertEqual(self.wf.build('x.png', 'style', 'negative', self.settings),
                         self.wf.build('x.png', 'style', 'negative', self.settings))

    def test_prefer_even_is_the_plain_draw(self):
        g = self.settings.grin
        g.aaa_min, g.aaa_max = -10.0, -20.0
        g.yaw_min, g.yaw_max = -15.0, 15.0
        self.settings.portrait.fixed_seed = True
        grin = self.wf.build('x.png', 'style', 'negative', self.settings)[self.wf.grin]['inputs']
        # one draw per value, in the order of the values
        rng = random.Random(FIXED_SEED)
        draws = dict(zip(GRIN_TARGETS + GRIN_ROTATIONS, (rng.random() for _ in range(12))))
        self.assertAlmostEqual(grin['aaa'], -10.0 - 10.0 * draws['aaa'])
        self.assertAlmostEqual(grin['rotate_yaw_min'], -15.0 + 30.0 * draws['yaw'])

    def test_prefer_shapes_the_draw(self):
        g = self.settings.grin
        g.aaa_min, g.aaa_max = -10.0, -20.0      # max is the lower number: prefer follows the fields
        g.yaw_min, g.yaw_max = -15.0, 15.0
        g.smile_min = g.smile_max = 2.0

        def shares(prefer: Prefer) -> tuple[float, float, float]:
            """Share of the picks within 10% of min, of max (aaa) and of either end (yaw)."""
            g.aaa_prefer = g.yaw_prefer = g.smile_prefer = prefer
            runs = 400
            near_min = near_max = near_end = 0
            for _ in range(runs):
                grin = self.wf.build('x.png', 'style', 'negative', self.settings)[self.wf.grin]['inputs']
                self.assertTrue(-20.0 <= grin['aaa'] <= -10.0)
                self.assertTrue(-15.0 <= grin['rotate_yaw_min'] <= 15.0)
                self.assertEqual(grin['smile'], 2.0)     # min == max: exactly that value, whatever the curve
                near_min += grin['aaa'] > -11.0
                near_max += grin['aaa'] < -19.0
                near_end += abs(grin['rotate_yaw_min']) > 12.0
            return near_min / runs, near_max / runs, near_end / runs

        even = shares(Prefer.EVEN)
        self.assertTrue(even[0] < 0.2 and even[1] < 0.2 and even[2] < 0.3)       # 10%, 10%, 20%
        self.assertGreater(shares(Prefer.MIN)[0], 0.22)                          # 32%
        self.assertGreater(shares(Prefer.MIN_STRONG)[0], 0.5)                    # 63%
        self.assertGreater(shares(Prefer.MAX)[1], 0.22)
        self.assertGreater(shares(Prefer.MAX_STRONG)[1], 0.5)
        self.assertGreater(shares(Prefer.ENDS)[2], 0.35)                         # 45%
        self.assertGreater(shares(Prefer.ENDS_STRONG)[2], 0.6)                   # 72%

    def test_still(self):
        g = self.settings.grin
        g.still = True
        g.aaa_min, g.aaa_max = -10.0, -20.0
        g.yaw_min, g.yaw_max = -15.0, 15.0
        g.aaa_prefer = g.yaw_prefer = Prefer.MIN_STRONG      # a still follows its position, not the curve
        # position 0 is the min field and 1 the max field, also when max is the lower number
        for position, aaa, yaw in ((0.0, -10.0, -15.0), (0.5, -15.0, 0.0), (1.0, -20.0, 15.0)):
            g.still_position = position
            built = self.wf.build('x.png', 'style', 'negative', self.settings)
            grin = built[self.wf.grin]['inputs']
            self.assertEqual((grin['aaa'], grin['rotate_yaw_min'], grin['rotate_yaw_max']), (aaa, yaw, yaw))
            self.assertEqual((grin['still'], grin['seed']), (True, FIXED_SEED))
            self.assertEqual(self.wf.frame_count(built), 1)
            self.assertEqual(built[self.wf.last_frame]['inputs']['batch_index'], 0)
        self.assertEqual(self.wf.build_replay(built, self.settings), built)      # unchanged: nothing to make again
        g.still = False
        animation = self.wf.build_replay(built, self.settings)
        self.assertEqual(self.wf.frame_count(animation), 120)
        self.assertEqual(animation[self.wf.last_frame]['inputs']['batch_index'], 119)

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

    def test_replay_only_changes_the_grin(self):
        self.settings.grin.smile_min = self.settings.grin.smile_max = 9.0
        self.settings.grin.frames = 60
        self.settings.portrait.steps = 12       # not a grin value: waits for the next capture
        replay = self.wf.build_replay(self.built, self.settings)
        grin = replay[self.wf.grin]['inputs']
        self.assertEqual((grin['smile'], grin['frames']), (9.0, 60))
        self.assertEqual(replay[self.wf.last_frame]['inputs']['batch_index'], 59)
        self.assertNotEqual(grin['seed'], self.built[self.wf.grin]['inputs']['seed'])
        grin_nodes = {self.wf.grin, self.wf.last_frame}
        for node_id, node in self.built.items():
            if node_id not in grin_nodes:
                self.assertEqual(replay[node_id], node, node_id)     # seeds included: ComfyUI's cache holds
        self.assertEqual(self.built[self.wf.grin]['inputs']['frames'], 120)     # the previous workflow is not changed

    def test_replay_fixed_seed_without_changes_is_identical(self):
        self.settings.portrait.fixed_seed = True
        first = self.wf.build('smile_capture.png', 'a style', 'a negative', self.settings)
        self.assertEqual(self.wf.build_replay(first, self.settings), first)
        self.settings.grin.smile_max = 9.0
        self.assertNotEqual(self.wf.build_replay(first, self.settings), first)

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
        # the capture's depth is the one the portrait is made with, not a second run
        self.assertEqual(self.built[self.wf.capture_depth]['inputs']['images'], self.built[self.wf.depth]['inputs']['image'])
        # both depths come from one model, and the last frame's from the last frame
        runs = [node for node in self.built.values() if node['class_type'] == 'DA3Inference']
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]['inputs']['da3_model'], runs[1]['inputs']['da3_model'])
        self.assertEqual(self.built['smile_last_geometry']['inputs']['image'], [self.wf.last_frame, 0])
        self.assertEqual(sum(node['class_type'] == 'LoadDA3Model' for node in self.built.values()), 1)

    def test_steps(self):
        steps = {node_id: self.wf.step(node_id) for node_id in self.built}
        self.assertEqual({name for name in steps.values() if name},
                         {'describe', 'depth', 'portrait', 'grin', 'depth of last frame'})
        self.assertEqual(steps[self.wf.sampler], 'portrait')
        self.assertEqual(self.built[self.wf.depth_run]['class_type'], 'DA3Inference')
        self.assertEqual(steps[self.wf.last_depth_run], 'depth of last frame')
        self.assertIsNone(steps[self.wf.load_image])
        self.assertIsNone(self.wf.step(None))

    def test_warmup_is_the_same_run_at_its_shortest(self):
        self.settings.portrait.fixed_seed = True
        full = self.wf.build('x.png', 'style', 'negative', self.settings)
        warmup = self.wf.build_warmup('x.png', 'style', 'negative', self.settings)
        self.assertEqual(warmup[self.wf.sampler]['inputs']['steps'], 1)
        self.assertEqual(self.wf.frame_count(warmup), 1)
        self.assertEqual(warmup[self.wf.last_frame]['inputs']['batch_index'], 0)
        # nothing else differs from a visitor's run
        warmup[self.wf.sampler]['inputs']['steps'] = full[self.wf.sampler]['inputs']['steps']
        warmup[self.wf.grin]['inputs']['still'] = False
        warmup[self.wf.last_frame]['inputs']['batch_index'] = full[self.wf.last_frame]['inputs']['batch_index']
        self.assertEqual(warmup, full)

    def test_template_not_changed_by_build(self):
        again = self.wf.build('other.png', 'other', 'other negative', self.settings)
        self.assertEqual(self.built[self.wf.load_image]['inputs']['image'], 'smile_capture.png')
        self.assertEqual(again[self.wf.load_image]['inputs']['image'], 'other.png')

    def test_grin_animates_the_whole_portrait(self):
        self.assertEqual(self.wf.portrait_size, (896, 1120))
        classes = {node['class_type'] for node in self.built.values()}
        self.assertNotIn('ImageCrop', classes)
        self.assertNotIn('ImageScale', classes)
        # the grin reads what the portrait preview shows: the decoded portrait, uncropped and unscaled
        self.assertEqual(self.built[self.wf.grin]['inputs']['image'], self.built[self.wf.portrait]['inputs']['images'])


class TestStyles(unittest.TestCase):
    def test_cycle(self):
        settings = PipelineSettings()
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
