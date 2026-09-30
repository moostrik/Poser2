"""The ComfyUI workflow (API format), adapted for SMILE and filled in per run.

SMILE crops and rotates the face itself (modules/face/FaceFinder.py), so the workflow's own
face crop is removed and its consumers read the uploaded image directly. The video and the
overlay check are removed too, and the saved portrait becomes a preview (nothing piles up in
ComfyUI's output folder). Nodes are found by class type or title, not by id.

The workflow only provides the structure: every run fills in the values from ComfySettings.
"""

import copy
import json
import random

from modules.comfy.settings import ComfySettings

MAX_SEED: int = 2 ** 32 - 1     # the smallest maximum of the seeded nodes (QwenVL: 1 .. 4294967295)
FIXED_SEED: int = 1             # with portrait.fixed_seed: the description and the grin too, or the runs still differ

# LoraSettings field: title of its node
LORAS: dict[str, str] = {'clothing': 'LoRA: clothing', 'looking': 'LoRA: looking at viewer', 'dark': 'LoRA: dark'}

GRIN_TARGETS: tuple[str, ...] = ('smile', 'aaa', 'eee', 'woo', 'eyebrow', 'blink', 'wink', 'pupil_x', 'pupil_y')
GRIN_ROTATIONS: tuple[str, ...] = ('pitch', 'yaw', 'roll')


def grin_box(portrait_size: tuple[int, int], crop_size: float) -> tuple[int, int, int, int]:
    """The centred 4:5 part of the portrait the grin shows: (x, y, width, height) in pixels."""
    portrait_width, portrait_height = portrait_size
    # a multiple of 8 keeps 4:5 exact and both margins even, so the crop is exactly centred
    width: int = max(8, int(round(portrait_width * crop_size / 8)) * 8)
    height: int = width * 5 // 4
    return (portrait_width - width) // 2, (portrait_height - height) // 2, width, height


def _find(workflow: dict, class_type: str | None = None, title: str | None = None) -> str:
    for node_id, node in workflow.items():
        if class_type is not None and node['class_type'] != class_type:
            continue
        if title is not None and node.get('_meta', {}).get('title') != title:
            continue
        return node_id
    raise KeyError(f'workflow has no node with class {class_type!r} and title {title!r}')


def _remove(workflow: dict, node_ids: set[str], replacement: list | None = None) -> None:
    """Remove nodes; links to them are pointed at *replacement* (or must not exist)."""
    for node_id in node_ids:
        workflow.pop(node_id, None)
    for node_id, node in workflow.items():
        for name, value in node['inputs'].items():
            if isinstance(value, list) and len(value) == 2 and value[0] in node_ids:
                if replacement is None:
                    raise ValueError(f'node {node_id} input {name!r} still uses removed node {value[0]}')
                node['inputs'][name] = list(replacement)


class Workflow():
    def __init__(self, path: str) -> None:
        with open(path, encoding='utf-8') as f:
            workflow: dict = json.load(f)

        self.load_image: str = _find(workflow, class_type='LoadImage')

        # the face crop chain: everything in the subgraph that holds the LivePortrait cropper
        cropper: str = _find(workflow, class_type='LivePortraitCropper')
        prefix: str = cropper.split(':')[0] + ':'
        crop_nodes: set[str] = {node_id for node_id in workflow if node_id.startswith(prefix)}
        _remove(workflow, crop_nodes, [self.load_image, 0])

        # outputs SMILE doesn't use
        unused: set[str] = {_find(workflow, class_type='VHS_VideoCombine'),
                            _find(workflow, title='Overlay'),
                            _find(workflow, title='Overlay preview')}
        _remove(workflow, unused)

        self.portrait: str = _find(workflow, title='Save portrait')
        workflow[self.portrait] = {'class_type': 'PreviewImage',
                                   'inputs': {'images': workflow[self.portrait]['inputs']['images']},
                                   '_meta': workflow[self.portrait]['_meta']}

        # SMILE streams the grin frames from the previews; writing them as 150 PNGs costs ~2 s,
        # but Grin Frames needs an output node to run at all: a text preview is the cheapest
        all_frames: str = _find(workflow, title='All frames')
        workflow[all_frames] = {'class_type': 'PreviewAny',
                                'inputs': {'source': workflow[all_frames]['inputs']['images']},
                                '_meta': workflow[all_frames]['_meta']}

        self.style: str = _find(workflow, title='Prompt: style')
        self.suffix: str = _find(workflow, title='Description + model')
        # the workflow joins description + suffix; a node in front of it adds the prefix
        self.prefix: str = 'smile_prefix'
        workflow[self.prefix] = {'class_type': 'StringConcatenate',
                                 'inputs': {'string_a': '', 'string_b': workflow[self.suffix]['inputs']['string_a'], 'delimiter': ''}}
        workflow[self.suffix]['inputs']['string_a'] = [self.prefix, 0]
        self.negative: str = _find(workflow, title='Negative')
        self.sampler: str = _find(workflow, class_type='KSampler')
        # KSamplerFullPreview (comfy/aiia_grin) previews every step with the real VAE; it needs the VAE
        self.vae: list = list(workflow[_find(workflow, class_type='VAEDecode')]['inputs']['vae'])
        self.loras: dict[str, str] = {name: _find(workflow, title=title) for name, title in LORAS.items()}
        self.instantid: str = _find(workflow, class_type='ApplyInstantIDAdvanced')
        self.depth: str = _find(workflow, class_type='ControlNetApplyAdvanced')
        self.describe: str = _find(workflow, class_type='AILab_QwenVL')
        self.description: str = _find(workflow, title='Preview as Text')
        self.grin_crop: str = _find(workflow, class_type='ImageCrop')
        self.grin_scale: str = _find(workflow, class_type='ImageScale')
        self.grin: str = _find(workflow, class_type='GrinFrames')

        # depth maps, as the old A1111 depth: of the capture (mask of the flash out) and of the
        # last grin frame (mask of the bloom). Part of this workflow on purpose: a second workflow
        # in between would drop this one from ComfyUI's cache and reload the models every visitor.
        self.capture_depth: str = 'smile_capture_depth'
        self.last_frame: str = 'smile_last_frame'
        self.last_depth: str = 'smile_last_depth'
        workflow['smile_capture_midas'] = _midas([self.load_image, 0])
        workflow[self.capture_depth] = {'class_type': 'PreviewImage', 'inputs': {'images': ['smile_capture_midas', 0]}}
        workflow[self.last_frame] = {'class_type': 'ImageFromBatch', 'inputs': {'image': [self.grin, 0], 'batch_index': 0, 'length': 1}}
        workflow['smile_last_midas'] = _midas([self.last_frame, 0])
        workflow[self.last_depth] = {'class_type': 'PreviewImage', 'inputs': {'images': ['smile_last_midas', 0]}}

        latent: dict = workflow[_find(workflow, class_type='EmptyLatentImage')]['inputs']
        self.portrait_size: tuple[int, int] = (latent['width'], latent['height'])
        self._template: dict = workflow

    def build(self, image_name: str, style: str, negative: str, s: ComfySettings) -> dict:
        workflow: dict = copy.deepcopy(self._template)
        workflow[self.load_image]['inputs']['image'] = image_name

        describe: dict = workflow[self.describe]['inputs']
        describe['custom_prompt'] = s.describe.instruction
        describe['max_tokens'] = s.describe.max_tokens

        p = s.portrait
        workflow[self.style]['inputs']['text'] = style
        # a space between the prefix and the description, nothing in front without a prefix
        prefix: str = s.describe.prefix.strip()
        workflow[self.prefix]['inputs'].update(string_a=prefix, delimiter=' ' if prefix else '')
        workflow[self.suffix]['inputs']['string_b'] = s.describe.suffix
        workflow[self.negative]['inputs']['text'] = negative
        sampler: dict = workflow[self.sampler]
        sampler['inputs'].update(steps=p.steps, cfg=p.cfg, sampler_name=p.sampler, scheduler=p.scheduler)
        if p.full_preview:
            sampler['class_type'] = 'KSamplerFullPreview'
            sampler['inputs']['vae'] = list(self.vae)
        workflow[self.instantid]['inputs'].update(ip_weight=p.embedding_strength, cn_strength=p.keypoint_strength)
        workflow[self.depth]['inputs'].update(strength=p.depth_strength, end_percent=p.depth_end)
        # one strength for the model and the text encoder, as A1111's <lora:name:strength>; 0 is off
        for name, node_id in self.loras.items():
            strength: float = getattr(s.lora, name)
            workflow[node_id]['inputs'].update(strength_model=strength, strength_clip=strength)

        g = s.grin
        x, y, width, height = grin_box(self.portrait_size, g.crop_size)
        workflow[self.grin_crop]['inputs'].update(x=x, y=y, width=width, height=height)
        workflow[self.grin_scale]['inputs'].update(width=g.width, height=g.width * 5 // 4)
        grin: dict = workflow[self.grin]['inputs']
        grin.update(frames=g.frames, ease=g.ease.value, crop_factor=g.crop_factor, midpoint=g.midpoint,
                    midpoint_deviation=g.midpoint_deviation, stitching=g.stitching)
        for name in GRIN_TARGETS:
            grin[name] = getattr(g, name)
        for name in GRIN_ROTATIONS:
            grin[f'rotate_{name}_min'] = getattr(g, f'{name}_min')
            grin[f'rotate_{name}_max'] = getattr(g, f'{name}_max')
        workflow[self.last_frame]['inputs']['batch_index'] = g.frames - 1

        for node_id in (self.sampler, self.describe, self.grin):
            workflow[node_id]['inputs']['seed'] = FIXED_SEED if p.fixed_seed else random.randint(1, MAX_SEED)
        return workflow


def _midas(image: list) -> dict:
    return {'class_type': 'MiDaS-DepthMapPreprocessor',
            'inputs': {'image': image, 'a': 6.283185307179586, 'bg_threshold': 0.1, 'resolution': 1024}}
