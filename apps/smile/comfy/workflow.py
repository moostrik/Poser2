"""The ComfyUI workflow (API format), adapted for SMILE and filled in per run.

SMILE crops and rotates the face itself (modules/face/FaceFinder.py), so the export starts from the
uploaded image; an older export's face crop subgraph and video recorder are removed and their consumers
rewired to the image. The overlay check is removed too, and the saved portrait becomes a preview (nothing
piles up in ComfyUI's output folder). The crop and scale in front of the grin are removed as well: the grin
animates the whole portrait. Nodes are found by class type or title, not by id.

The workflow only provides the structure: every run fills in the values from PipelineSettings.
"""

import copy
import json
import logging
import random
from typing import Callable

import pytweening as easing

from apps.smile.comfy.settings import PipelineSettings, Prefer, StyleSettings

logger = logging.getLogger(__name__)

MAX_SEED: int = 2 ** 32 - 1     # the smallest maximum of the seeded nodes (QwenVL: 1 .. 4294967295)
FIXED_SEED: int = 1             # with portrait.fixed_seed: the description and the grin too, or the runs still differ

LORA_SLOTS: int = 3     # StyleSettings lora_1 .. lora_3: the workflow's LoRA nodes, in the order of their chain
NO_LORA: str = 'none'

GRIN_TARGETS: tuple[str, ...] = ('smile', 'aaa', 'eee', 'woo', 'eyebrow', 'blink', 'wink', 'pupil_x', 'pupil_y')
GRIN_ROTATIONS: tuple[str, ...] = ('pitch', 'yaw', 'roll')

# a curve over the random draw (0..1) of a grin value: where the curve is flat the picks bunch up,
# so an ease-in prefers min, an ease-out max and an in-out both ends; quint pulls harder than quad
PREFER_CURVES: dict[Prefer, Callable[[float], float]] = {
    Prefer.EVEN:        easing.linear,
    Prefer.MIN:         easing.easeInQuad,
    Prefer.MIN_STRONG:  easing.easeInQuint,
    Prefer.MAX:         easing.easeOutQuad,
    Prefer.MAX_STRONG:  easing.easeOutQuint,
    Prefer.ENDS:        easing.easeInOutQuad,
    Prefer.ENDS_STRONG: easing.easeInOutQuint,
}


def _find(workflow: dict, class_type: str | None = None, title: str | None = None) -> str:
    node_id: str | None = _find_optional(workflow, class_type, title)
    if node_id is None:
        raise KeyError(f'workflow has no node with class {class_type!r} and title {title!r}')
    return node_id


def _find_optional(workflow: dict, class_type: str | None = None, title: str | None = None) -> str | None:
    for node_id, node in workflow.items():
        if class_type is not None and node['class_type'] != class_type:
            continue
        if title is not None and node.get('_meta', {}).get('title') != title:
            continue
        return node_id
    return None


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


def _lora_chain(workflow: dict) -> list[str]:
    """The LoRA nodes in the order the model passes through them, from the checkpoint on."""
    loras: dict[str, str] = {node['inputs']['model'][0]: node_id for node_id, node in workflow.items()
                             if node['class_type'] == 'LoraLoader'}      # source node: the LoRA node it feeds
    chain: list[str] = []
    source: str = _find(workflow, class_type='CheckpointLoaderSimple')
    while source in loras:
        source = loras[source]
        chain.append(source)
    if len(chain) != LORA_SLOTS or len(loras) != LORA_SLOTS:
        raise ValueError(f'workflow needs {LORA_SLOTS} LoraLoader nodes in a row after the checkpoint, found {len(chain)} of {len(loras)}')
    return chain


class Workflow():
    def __init__(self, path: str) -> None:
        with open(path, encoding='utf-8') as f:
            workflow: dict = json.load(f)

        self.load_image: str = _find(workflow, class_type='LoadImage')

        # an older export's face crop chain (the subgraph that holds the LivePortrait cropper) and video recorder
        cropper: str | None = _find_optional(workflow, class_type='LivePortraitCropper')
        if cropper is not None:
            prefix: str = cropper.split(':')[0] + ':'
            _remove(workflow, {node_id for node_id in workflow if node_id.startswith(prefix)}, [self.load_image, 0])
        video: str | None = _find_optional(workflow, class_type='VHS_VideoCombine')
        if video is not None:
            _remove(workflow, {video})

        # outputs SMILE doesn't use
        _remove(workflow, {_find(workflow, title='Overlay'), _find(workflow, title='Overlay preview')})

        self.portrait: str = _find(workflow, title='Save portrait')
        workflow[self.portrait] = {'class_type': 'PreviewImage',
                                   'inputs': {'images': workflow[self.portrait]['inputs']['images']},
                                   '_meta': workflow[self.portrait]['_meta']}

        self.portrait_preview: str = _find(workflow, title='Portrait')     # the workflow's own preview of the portrait

        # SMILE streams the grin frames from the previews; writing them as 150 PNGs costs ~2 s,
        # but Grin Frames needs an output node to run at all: a text preview is the cheapest
        self.all_frames: str = _find(workflow, title='All frames')
        workflow[self.all_frames] = {'class_type': 'PreviewAny',
                                     'inputs': {'source': workflow[self.all_frames]['inputs']['images']},
                                     '_meta': workflow[self.all_frames]['_meta']}

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
        self.checkpoint: str = _find(workflow, class_type='CheckpointLoaderSimple')
        self.loras: list[str] = _lora_chain(workflow)
        # the distillation LoRA (portrait.distill_lora): a node build() puts between the checkpoint and the style LoRAs
        self.distill: str = 'smile_distill'
        self.instantid: str = _find(workflow, class_type='ApplyInstantIDAdvanced')
        self.depth: str = _find(workflow, class_type='ControlNetApplyAdvanced')
        self.describe: str = _find(workflow, class_type='AILab_QwenVL')
        # the basic node fixes repetition_penalty at 1.2, which drops the commas; the advanced one takes it
        workflow[self.describe]['class_type'] = 'AILab_QwenVL_Advanced'
        workflow[self.describe]['inputs'].update(use_torch_compile=False, device='auto', temperature=0.6,
                                                 top_p=0.9, num_beams=1, repetition_penalty=1.0, frame_count=16,
                                                 video_frame_size='auto')     # required since QwenVL 2.3 (video input only; we send an image)
        self.description: str = _find(workflow, title='Preview as Text')
        self.grin: str = _find(workflow, class_type='GrinFrames')

        # the grin animates the whole portrait at its own size: FaceFinder already sets the framing
        grin_crop: str = _find(workflow, class_type='ImageCrop')
        grin_scale: str = _find(workflow, class_type='ImageScale')
        _remove(workflow, {grin_crop, grin_scale}, workflow[grin_crop]['inputs']['image'])

        # depth maps, as the old A1111 depth: of the capture (mask of the flash out) and of the
        # last grin frame (mask of the bloom). Part of this workflow on purpose: a second workflow
        # in between would drop this one from ComfyUI's cache and reload the models every visitor.
        # The capture's depth is the one the portrait is made with; the last frame gets copies of the
        # workflow's two depth nodes, on the same model.
        self.capture_depth: str = 'smile_capture_depth'
        self.last_frame: str = 'smile_last_frame'
        self.last_depth: str = 'smile_last_depth'
        depth_map: str = workflow[self.depth]['inputs']['image'][0]
        self.depth_run: str = workflow[depth_map]['inputs']['da3_geometry'][0]
        self.last_depth_run: str = 'smile_last_geometry'
        workflow[self.capture_depth] = {'class_type': 'PreviewImage', 'inputs': {'images': [depth_map, 0]}}
        workflow[self.last_frame] = {'class_type': 'ImageFromBatch', 'inputs': {'image': [self.grin, 0], 'batch_index': 0, 'length': 1}}
        workflow[self.last_depth_run] = copy.deepcopy(workflow[self.depth_run])
        workflow[self.last_depth_run]['inputs']['image'] = [self.last_frame, 0]
        workflow['smile_last_depth_map'] = copy.deepcopy(workflow[depth_map])
        workflow['smile_last_depth_map']['inputs']['da3_geometry'] = [self.last_depth_run, 0]
        workflow[self.last_depth] = {'class_type': 'PreviewImage', 'inputs': {'images': ['smile_last_depth_map', 0]}}

        latent: dict = workflow[_find(workflow, class_type='EmptyLatentImage')]['inputs']
        self.portrait_size: tuple[int, int] = (latent['width'], latent['height'])
        self._template: dict = workflow
        self._warned: set[str] = set()      # model files ComfyUI does not offer, each said once
        # the nodes that take time, by the name the panel shows while they run
        self._steps: dict[str, str] = {self.describe: 'describe', self.depth_run: 'depth', self.sampler: 'portrait',
                                       self.grin: 'grin', self.last_depth_run: 'depth of last frame'}

    def step(self, node_id: str | None) -> str | None:
        """The step of the generation a node is, or None for the nodes in between."""
        return self._steps.get(node_id) if node_id is not None else None

    @property
    def step_count(self) -> int:
        """The steps a full run goes through."""
        return len(self._steps)

    def build(self, image_name: str, style: StyleSettings, s: PipelineSettings) -> dict:
        """The workflow of one visitor, in one of the styles of s (style_a, style_b)."""
        workflow: dict = copy.deepcopy(self._template)
        workflow[self.load_image]['inputs']['image'] = image_name

        describe: dict = workflow[self.describe]['inputs']
        describe['custom_prompt'] = style.instruction
        describe['max_tokens'] = style.max_tokens
        describe['repetition_penalty'] = style.repetition_penalty

        p = s.portrait
        workflow[self.style]['inputs']['text'] = style.prompt
        # a space between the prefix and the description, nothing in front without a prefix
        prefix: str = style.prefix.strip()
        workflow[self.prefix]['inputs'].update(string_a=prefix, delimiter=' ' if prefix else '')
        workflow[self.suffix]['inputs']['string_b'] = style.suffix
        workflow[self.negative]['inputs']['text'] = style.negative
        sampler: dict = workflow[self.sampler]
        sampler['inputs'].update(steps=p.steps, cfg=p.cfg, sampler_name=p.sampler, scheduler=p.scheduler)
        if p.full_preview:
            sampler['class_type'] = 'KSamplerFullPreview'
            sampler['inputs']['vae'] = list(self.vae)
        workflow[self.instantid]['inputs'].update(ip_weight=p.embedding_strength, cn_strength=p.keypoint_strength)
        workflow[self.depth]['inputs'].update(strength=p.depth_strength, end_percent=p.depth_end)
        # ComfyUI refuses a file it doesn't have: the workflow's checkpoint stays, and no distillation.
        # An install that misses a model would otherwise run quietly with another: say it, once per file
        if p.checkpoint in p.checkpoint_options:
            workflow[self.checkpoint]['inputs']['ckpt_name'] = p.checkpoint
        else:
            self._missing('checkpoint', p.checkpoint, f"the workflow's {workflow[self.checkpoint]['inputs']['ckpt_name']}")
        if p.distill_lora != NO_LORA and p.distill_strength != 0.0 and p.distill_lora not in p.lora_options:
            self._missing('distillation LoRA', p.distill_lora, 'no distillation')
        if p.distill_lora != NO_LORA and p.distill_lora in p.lora_options and p.distill_strength != 0.0:
            # on the model only (the text encoder stays the checkpoint's), in front of the style LoRAs
            workflow[self.distill] = {'class_type': 'LoraLoaderModelOnly',
                                      'inputs': {'model': [self.checkpoint, 0], 'lora_name': p.distill_lora,
                                                 'strength_model': p.distill_strength}}
            workflow[self.loras[0]]['inputs']['model'] = [self.distill, 0]
        # one strength for the model and the text encoder, as A1111's <lora:name:strength>; 0 is off
        for slot, node_id in enumerate(self.loras, start=1):
            name: str = getattr(style, f'lora_{slot}')
            strength: float = getattr(style, f'lora_{slot}_strength')
            if name != NO_LORA and name not in style.lora_options:
                self._missing(f'LoRA {slot}', name, 'the slot switched off')
            if name == NO_LORA or name not in style.lora_options:
                # ComfyUI refuses a file it doesn't have: the slot keeps the workflow's file, switched off
                name, strength = workflow[node_id]['inputs']['lora_name'], 0.0
            workflow[node_id]['inputs'].update(lora_name=name, strength_model=strength, strength_clip=strength)

        # one seed per visitor: the portrait, the description and the grin draw from generators of their own
        seed: int = self._seed(s)
        for node_id in (self.sampler, self.describe):
            workflow[node_id]['inputs']['seed'] = seed
        self._fill_grin(workflow, s, seed)
        return workflow

    def _missing(self, what: str, name: str, instead: str) -> None:
        if name not in self._warned:
            self._warned.add(name)
            logger.warning('ComfyUI does not offer the %s %s: %s is used instead', what, name, instead)

    def build_warmup(self, image_name: str, style: StyleSettings, s: PipelineSettings) -> dict:
        """The same run at its shortest: one sampler step and one grin frame. The models load at the
        first step and the first frame; the rest of a full run would only cost time."""
        workflow: dict = self.build(image_name, style, s)
        workflow[self.sampler]['inputs']['steps'] = 1
        workflow[self.grin]['inputs']['still'] = True
        workflow[self.last_frame]['inputs']['batch_index'] = 0
        return workflow

    def build_portrait(self, image_name: str, style: StyleSettings, s: PipelineSettings) -> dict:
        """The same run without the grin (skip_animation): the grin's outputs are removed, so
        ComfyUI stops after the portrait. The other nodes stay, so ComfyUI keeps their cache."""
        workflow: dict = self.build(image_name, style, s)
        for node_id in (self.all_frames, self.last_depth):
            del workflow[node_id]
        return workflow

    def build_describe(self, image_name: str, style: StyleSettings, s: PipelineSettings) -> dict:
        """The same run without the portrait and the grin (skip_generation): only the description
        and the capture's depth are left as outputs."""
        workflow: dict = self.build_portrait(image_name, style, s)
        for node_id in (self.portrait, self.portrait_preview):
            del workflow[node_id]
        return workflow

    def build_replay(self, previous: dict, s: PipelineSettings) -> dict:
        """The previous workflow with only the grin filled in again (the grin loop). Everything up to
        the portrait is unchanged, so ComfyUI takes it from its cache and only makes the grin."""
        workflow: dict = copy.deepcopy(previous)
        self._fill_grin(workflow, s)
        return workflow

    def run_facts(self, workflow: dict) -> dict:
        """What only this run knows, read back from its workflow: the visitor's seed, and of every grin
        value its peak and the moment of that peak (rounded: this is for the visitor's record)."""
        grin: dict = workflow[self.grin]['inputs']
        peak: dict[str, float] = {name: grin[name] for name in GRIN_TARGETS}
        peak.update({name: grin[f'rotate_{name}_min'] for name in GRIN_ROTATIONS})
        peak_at: dict[str, float] = {name: grin[f'{name}_peak'] for name in GRIN_TARGETS}
        peak_at.update({name: grin[f'rotate_{name}_peak'] for name in GRIN_ROTATIONS})
        return {'seed': workflow[self.sampler]['inputs']['seed'],
                'peak': {name: round(value, 2) for name, value in peak.items()},
                'peak_at': {name: round(value, 3) for name, value in peak_at.items()}}

    def _fill_grin(self, workflow: dict, s: PipelineSettings, seed: int | None = None) -> None:
        """seed is the visitor's; without one (a replay) the grin draws its own."""
        g = s.grin
        grin: dict = workflow[self.grin]['inputs']
        grin.update(frames=g.frames, ease=g.ease.value, crop_factor=g.face_region, midpoint=g.midpoint,
                    midpoint_deviation=g.midpoint_deviation, stitching=g.stitching)

        # a still has nothing random left, so its seed is fixed: unchanged settings give an identical workflow
        still: bool = g.still
        if still:
            seed = FIXED_SEED
        elif seed is None:
            seed = self._seed(s)
        rng: random.Random = random.Random(seed)

        def peak(name: str) -> float:
            """This pass's peak of one value: its place between min and max for a still, else a random draw
            shaped by its prefer curve. The two ends are used as set (min may be the higher number), so
            still position 1 is the max field."""
            low, high = getattr(g, f'{name}_min'), getattr(g, f'{name}_max')
            position: float = g.still_position if still else PREFER_CURVES[getattr(g, f'{name}_prefer')](rng.random())
            return low + position * (high - low)

        # SMILE picks the pose and the node animates to it, so the node's rotation ranges hold one value
        for name in GRIN_TARGETS:
            grin[name] = peak(name)
        for name in GRIN_ROTATIONS:
            grin[f'rotate_{name}_min'] = grin[f'rotate_{name}_max'] = peak(name)

        # the moment of each peak is SMILE's pick too (the node would draw its own), so the record has it
        def moment() -> float:
            return min(max(g.midpoint + rng.uniform(-g.midpoint_deviation, g.midpoint_deviation), 0.01), 0.99)

        for name in GRIN_TARGETS:
            grin[f'{name}_peak'] = moment()
        for name in GRIN_ROTATIONS:
            grin[f'rotate_{name}_peak'] = moment()
        grin.update(seed=seed, still=still)
        workflow[self.last_frame]['inputs']['batch_index'] = 0 if still else g.frames - 1

    def frame_count(self, workflow: dict) -> int:
        """Grin frames this workflow makes: one for a still."""
        grin: dict = workflow[self.grin]['inputs']
        return 1 if grin['still'] else grin['frames']

    @staticmethod
    def _seed(s: PipelineSettings) -> int:
        return FIXED_SEED if s.portrait.fixed_seed else random.randint(1, MAX_SEED)
