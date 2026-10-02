"""Tests for adapting the exported ComfyUI workflow (no ComfyUI server needed)."""

import os
import random
import unittest

from modules.comfy.settings import PipelineSettings, Ease, Prefer, StyleMode
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
        self.settings.style_a.prompt = 'a style'
        self.settings.style_a.negative = 'a negative'
        self.built = self.wf.build('smile_capture.png', self.settings.style_a, self.settings)

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
        s.style_b.max_tokens = 200
        s.style_b.repetition_penalty = 1.1
        s.style_b.instruction = 'describe'
        s.style_b.prefix = 'prefix'
        s.style_b.suffix = ' suffix'
        s.style_b.prompt = 'style'
        s.style_b.negative = 'negative'
        s.portrait.steps = 8
        s.portrait.cfg = 2.0
        s.portrait.sampler = 'euler'
        s.portrait.scheduler = 'karras'
        s.portrait.embedding_strength = 0.8
        s.portrait.keypoint_strength = 0.7
        s.portrait.depth_strength = 0.6
        s.portrait.depth_end = 0.4
        s.style_b.lora_options = ['none', 'clothing_slider.safetensors', 'looking_at_viewer.safetensors', 'dark.safetensors', 'other.safetensors']
        s.style_b.lora_1, s.style_b.lora_1_strength = 'other.safetensors', 0.9
        s.style_b.lora_2, s.style_b.lora_2_strength = 'none', 2.0
        s.style_b.lora_3, s.style_b.lora_3_strength = 'missing.safetensors', -0.3
        s.grin.ease = Ease.LINEAR
        s.grin.smile_min = s.grin.smile_max = 3.0
        s.grin.yaw_min = s.grin.yaw_max = 10.0
        s.grin.stitching = False
        b = self.wf.build('x.png', s.style_b, s)
        describe = b[self.wf.describe]['inputs']
        self.assertEqual((describe['custom_prompt'], describe['max_tokens']), ('describe', 200))
        # the other style has its own describe settings
        other = self.wf.build('x.png', s.style_a, s)[self.wf.describe]['inputs']
        self.assertEqual((other['custom_prompt'], other['max_tokens'], other['repetition_penalty']), (s.style_a.instruction, 120, 1.0))
        self.assertNotEqual(s.style_a.instruction, 'describe')
        self.assertEqual((b[self.wf.describe]['class_type'], describe['repetition_penalty']), ('AILab_QwenVL_Advanced', 1.1))
        # prefix + space + description, then + suffix
        prefix = b[self.wf.prefix]['inputs']
        self.assertEqual((prefix['string_a'], prefix['delimiter'], prefix['string_b']), ('prefix', ' ', [self.wf.describe, 0]))
        self.assertEqual(self.built[self.wf.prefix]['inputs']['delimiter'], '')     # no prefix: no leading space
        suffix = b[self.wf.suffix]['inputs']
        self.assertEqual((suffix['string_a'], suffix['string_b']), ([self.wf.prefix, 0], ' suffix'))
        self.assertEqual((b[self.wf.style]['inputs']['text'], b[self.wf.negative]['inputs']['text']), ('style', 'negative'))
        sampler = b[self.wf.sampler]['inputs']
        self.assertEqual((sampler['steps'], sampler['cfg'], sampler['sampler_name'], sampler['scheduler']), (8, 2.0, 'euler', 'karras'))
        instantid = b[self.wf.instantid]['inputs']
        self.assertEqual((instantid['ip_weight'], instantid['cn_strength']), (0.8, 0.7))
        depth = b[self.wf.depth]['inputs']
        self.assertEqual((depth['strength'], depth['end_percent']), (0.6, 0.4))
        # a slot on none, or on a file ComfyUI doesn't have, keeps the workflow's file, switched off
        slots = (('other.safetensors', 0.9), ('looking_at_viewer.safetensors', 0.0), ('dark.safetensors', 0.0))
        for node_id, (name, strength) in zip(self.wf.loras, slots):
            lora = b[node_id]['inputs']
            self.assertEqual((lora['lora_name'], lora['strength_model'], lora['strength_clip']), (name, strength, strength))
        # the other style has its own LoRAs
        slots = (('clothing_slider.safetensors', 1.0), ('looking_at_viewer.safetensors', 1.0), ('dark.safetensors', -0.5))
        for node_id, (name, strength) in zip(self.wf.loras, slots):
            lora = self.wf.build('x.png', s.style_a, s)[node_id]['inputs']
            self.assertEqual((lora['lora_name'], lora['strength_model'], lora['strength_clip']), (name, strength, strength))
        grin = b[self.wf.grin]['inputs']
        self.assertEqual((grin['ease'], grin['smile'], grin['stitching']), ('linear', 3.0, False))
        self.assertEqual((grin['rotate_yaw_min'], grin['rotate_yaw_max']), (10.0, 10.0))

    def test_checkpoint(self):
        p = self.settings.portrait
        default = self.built[self.wf.checkpoint]['inputs']['ckpt_name']
        p.checkpoint_options = [default, 'other.safetensors']
        p.checkpoint = 'other.safetensors'
        b = self.wf.build('x.png', self.settings.style_a, self.settings)
        self.assertEqual(b[self.wf.checkpoint]['inputs']['ckpt_name'], 'other.safetensors')
        p.checkpoint = 'missing.safetensors'        # a file ComfyUI doesn't have: the workflow's stays
        b = self.wf.build('x.png', self.settings.style_a, self.settings)
        self.assertEqual(b[self.wf.checkpoint]['inputs']['ckpt_name'], default)

    def test_distill_lora(self):
        p = self.settings.portrait
        first = self.built[self.wf.loras[0]]['inputs']
        self.assertNotIn(self.wf.distill, self.built)
        self.assertEqual(first['model'], [self.wf.checkpoint, 0])
        p.lora_options = ['none', 'dmd2.safetensors']
        p.distill_lora, p.distill_strength = 'dmd2.safetensors', 0.7
        b = self.wf.build('x.png', self.settings.style_a, self.settings)
        self.assertEqual(b[self.wf.distill], {'class_type': 'LoraLoaderModelOnly',
                                              'inputs': {'model': [self.wf.checkpoint, 0], 'lora_name': 'dmd2.safetensors', 'strength_model': 0.7}})
        # the model goes through it to the style LoRAs, the text encoder stays the checkpoint's
        self.assertEqual(b[self.wf.loras[0]]['inputs']['model'], [self.wf.distill, 0])
        self.assertEqual(b[self.wf.loras[0]]['inputs']['clip'], [self.wf.checkpoint, 1])
        # off: none, strength 0, or a file ComfyUI doesn't have
        for name, strength in (('none', 0.7), ('dmd2.safetensors', 0.0), ('missing.safetensors', 0.7)):
            p.distill_lora, p.distill_strength = name, strength
            b = self.wf.build('x.png', self.settings.style_a, self.settings)
            self.assertNotIn(self.wf.distill, b)
            self.assertEqual(b[self.wf.loras[0]]['inputs']['model'], [self.wf.checkpoint, 0])

    def test_peaks_random_within_the_range(self):
        g = self.settings.grin
        g.aaa_min, g.aaa_max = -10.0, -20.0      # max may be the lower number
        g.yaw_min, g.yaw_max = -15.0, 15.0
        peaks = [self.wf.build('x.png', self.settings.style_a, self.settings)[self.wf.grin]['inputs'] for _ in range(20)]
        for grin in peaks:
            self.assertTrue(-20.0 <= grin['aaa'] <= -10.0)
            self.assertTrue(-15.0 <= grin['rotate_yaw_min'] <= 15.0)
            self.assertEqual(grin['rotate_yaw_min'], grin['rotate_yaw_max'])     # the node gets the pick, not the range
            self.assertEqual(grin['smile'], 2.0)                                 # min == max: no variation
            self.assertFalse(grin['still'])
        self.assertGreater(len({grin['aaa'] for grin in peaks}), 1)
        self.settings.portrait.fixed_seed = True
        self.assertEqual(self.wf.build('x.png', self.settings.style_a, self.settings),
                         self.wf.build('x.png', self.settings.style_a, self.settings))

    def test_prefer_even_is_the_plain_draw(self):
        g = self.settings.grin
        g.aaa_min, g.aaa_max = -10.0, -20.0
        g.yaw_min, g.yaw_max = -15.0, 15.0
        self.settings.portrait.fixed_seed = True
        grin = self.wf.build('x.png', self.settings.style_a, self.settings)[self.wf.grin]['inputs']
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
                grin = self.wf.build('x.png', self.settings.style_a, self.settings)[self.wf.grin]['inputs']
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
            built = self.wf.build('x.png', self.settings.style_a, self.settings)
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
        other = self.wf.build('smile_capture.png', self.settings.style_a, self.settings)
        for node_id in (self.wf.sampler, self.wf.describe, self.wf.grin):
            seed = self.built[node_id]['inputs']['seed']
            self.assertTrue(1 <= seed <= MAX_SEED)
            self.assertNotEqual(seed, other[node_id]['inputs']['seed'])

    def test_one_seed_per_visitor(self):
        seeds = {self.built[node_id]['inputs']['seed'] for node_id in (self.wf.sampler, self.wf.describe, self.wf.grin)}
        self.assertEqual(len(seeds), 1)

    def test_run_facts(self):
        g = self.settings.grin
        g.aaa_min, g.aaa_max = -10.0, -20.0
        g.yaw_min, g.yaw_max = -15.0, 15.0
        built = self.wf.build('x.png', self.settings.style_a, self.settings)
        facts = self.wf.run_facts(built)
        grin = built[self.wf.grin]['inputs']
        self.assertEqual(facts['seed'], built[self.wf.sampler]['inputs']['seed'])
        for block in ('peak', 'peak_at'):
            self.assertEqual(set(facts[block]), set(GRIN_TARGETS + GRIN_ROTATIONS))
        self.assertEqual((facts['peak']['aaa'], facts['peak']['yaw']), (round(grin['aaa'], 2), round(grin['rotate_yaw_min'], 2)))
        self.assertEqual((facts['peak_at']['aaa'], facts['peak_at']['yaw']),
                         (round(grin['aaa_peak'], 3), round(grin['rotate_yaw_peak'], 3)))

    def test_peak_moments_are_sent_to_the_node(self):
        g = self.settings.grin
        g.midpoint, g.midpoint_deviation = 0.9, 0.3
        names = [f'{name}_peak' for name in GRIN_TARGETS] + [f'rotate_{name}_peak' for name in GRIN_ROTATIONS]
        runs = [self.wf.build('x.png', self.settings.style_a, self.settings)[self.wf.grin]['inputs'] for _ in range(20)]
        for grin in runs:
            for name in names:
                self.assertTrue(0.6 <= grin[name] <= 0.99, name)       # midpoint +- deviation, kept inside the loop
        self.assertGreater(len({grin['smile_peak'] for grin in runs}), 1)
        self.assertGreater(len({runs[0][name] for name in names}), 1)  # each value its own moment
        g.midpoint_deviation = 0.0
        grin = self.wf.build('x.png', self.settings.style_a, self.settings)[self.wf.grin]['inputs']
        self.assertEqual({grin[name] for name in names}, {0.9})

    def test_fixed_seed(self):
        self.settings.portrait.fixed_seed = True
        first = self.wf.build('smile_capture.png', self.settings.style_a, self.settings)
        second = self.wf.build('smile_capture.png', self.settings.style_a, self.settings)
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
        first = self.wf.build('smile_capture.png', self.settings.style_a, self.settings)
        self.assertEqual(self.wf.build_replay(first, self.settings), first)
        self.settings.grin.smile_max = 9.0
        self.assertNotEqual(self.wf.build_replay(first, self.settings), first)

    def test_full_preview_sampler(self):
        sampler = self.built[self.wf.sampler]
        self.assertEqual(sampler['class_type'], 'KSamplerFullPreview')
        decode = next(node for node in self.built.values() if node['class_type'] == 'VAEDecode')
        self.assertEqual(sampler['inputs']['vae'], decode['inputs']['vae'])
        self.settings.portrait.full_preview = False
        plain = self.wf.build('x.png', self.settings.style_a, self.settings)[self.wf.sampler]
        self.assertEqual(plain['class_type'], 'KSampler')
        self.assertNotIn('vae', plain['inputs'])

    def test_lora_chain(self):
        # checkpoint -> the LoRAs in a row -> InstantID (model) and the three text encoders (clip)
        checkpoint = next(node_id for node_id, node in self.built.items() if node['class_type'] == 'CheckpointLoaderSimple')
        source = checkpoint
        self.assertEqual(len(self.wf.loras), 3)
        for node_id in self.wf.loras:
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
        full = self.wf.build('x.png', self.settings.style_a, self.settings)
        warmup = self.wf.build_warmup('x.png', self.settings.style_a, self.settings)
        self.assertEqual(warmup[self.wf.sampler]['inputs']['steps'], 1)
        self.assertEqual(self.wf.frame_count(warmup), 1)
        self.assertEqual(warmup[self.wf.last_frame]['inputs']['batch_index'], 0)
        # nothing else differs from a visitor's run
        warmup[self.wf.sampler]['inputs']['steps'] = full[self.wf.sampler]['inputs']['steps']
        warmup[self.wf.grin]['inputs']['still'] = False
        warmup[self.wf.last_frame]['inputs']['batch_index'] = full[self.wf.last_frame]['inputs']['batch_index']
        self.assertEqual(warmup, full)

    def test_describe_only_runs_the_description_and_the_capture_depth(self):
        self.settings.portrait.fixed_seed = True
        full = self.wf.build('x.png', self.settings.style_a, self.settings)
        short = self.wf.build_describe('x.png', self.settings.style_a, self.settings)
        removed = {self.wf.portrait, self.wf.portrait_preview, self.wf.all_frames, self.wf.last_depth}
        self.assertEqual(set(full) - set(short), removed)
        # nothing else differs from a visitor's run, so ComfyUI's cache holds
        for node_id, node in short.items():
            self.assertEqual(node, full[node_id], node_id)
        for node_id, name, source in links(short):
            self.assertIn(source, short, f'{node_id}.{name} -> missing {source}')
        # ComfyUI runs what its output nodes need: the description and the capture's depth
        outputs = [node_id for node_id, node in short.items() if node['class_type'] in ('PreviewImage', 'PreviewAny', 'SaveImage')]
        self.assertTrue({self.wf.capture_depth, self.wf.description} <= set(outputs))
        needed, todo = set(), outputs
        while todo:
            node_id = todo.pop()
            if node_id not in needed:
                needed.add(node_id)
                todo += [value[0] for value in short[node_id]['inputs'].values() if isinstance(value, list) and len(value) == 2]
        self.assertTrue({self.wf.describe, self.wf.depth_run} <= needed)
        self.assertFalse({self.wf.sampler, self.wf.grin, self.wf.last_depth_run} & needed)

    def test_portrait_only_leaves_out_the_grin(self):
        self.settings.portrait.fixed_seed = True
        full = self.wf.build('x.png', self.settings.style_a, self.settings)
        short = self.wf.build_portrait('x.png', self.settings.style_a, self.settings)
        self.assertEqual(set(full) - set(short), {self.wf.all_frames, self.wf.last_depth})
        for node_id, node in short.items():
            self.assertEqual(node, full[node_id], node_id)
        outputs = [node_id for node_id, node in short.items() if node['class_type'] in ('PreviewImage', 'PreviewAny', 'SaveImage')]
        self.assertTrue({self.wf.capture_depth, self.wf.description, self.wf.portrait} <= set(outputs))
        needed, todo = set(), outputs
        while todo:
            node_id = todo.pop()
            if node_id not in needed:
                needed.add(node_id)
                todo += [value[0] for value in short[node_id]['inputs'].values() if isinstance(value, list) and len(value) == 2]
        self.assertTrue({self.wf.describe, self.wf.depth_run, self.wf.sampler} <= needed)
        self.assertFalse({self.wf.grin, self.wf.last_depth_run} & needed)

    def test_template_not_changed_by_build(self):
        again = self.wf.build('other.png', self.settings.style_b, self.settings)
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
        settings.style_visitors_a = settings.style_visitors_b = 2
        styles = Styles(settings)
        ids = [styles.advance() for _ in range(3)]
        self.assertEqual(ids, ['0000', '0000', '0001'])
        self.assertEqual(settings.style_current, 'B')
        self.assertIs(styles.current(), settings.style_b)
        ids = [styles.advance() for _ in range(2)]
        self.assertEqual(ids, ['0001', '0000'])
        self.assertEqual(settings.style_current, 'A')
        self.assertIs(styles.current(), settings.style_a)

    def test_each_style_has_its_own_number_of_visitors(self):
        settings = PipelineSettings()
        settings.style_visitors_a = 3
        settings.style_visitors_b = 1
        styles = Styles(settings)
        ids = [styles.advance() for _ in range(9)]
        self.assertEqual(ids, ['0000', '0000', '0000', '0001', '0000', '0000', '0000', '0001', '0000'])

    def test_one_style_for_everyone(self):
        settings = PipelineSettings()
        settings.style_visitors_a = settings.style_visitors_b = 2
        styles = Styles(settings)
        settings.style_mode = StyleMode.B
        self.assertEqual([styles.advance() for _ in range(5)], ['0001'] * 5)
        self.assertEqual(settings.style_current, 'B')
        self.assertIs(styles.current(), settings.style_b)
        settings.style_mode = StyleMode.A
        self.assertEqual([styles.advance() for _ in range(3)], ['0000'] * 3)
        self.assertEqual(settings.style_current, 'A')
        # back on both: a full run of the style that was forced, then the other
        settings.style_mode = StyleMode.BOTH
        self.assertEqual([styles.advance() for _ in range(3)], ['0000', '0000', '0001'])


if __name__ == '__main__':
    unittest.main()
