"""The ComfyUI workflow (API format), adapted for SMILE and filled in per run.

SMILE crops and rotates the face itself (modules/face/FaceFinder.py), so the workflow's own
face crop is removed and its consumers read the uploaded image directly. The video and the
overlay check are removed too, and the saved portrait becomes a preview (nothing piles up in
ComfyUI's output folder). The crop and scale in front of the grin are removed as well: the grin
animates the whole portrait. Nodes are found by class type or title, not by id.

The workflow only provides the structure: every run fills in the values from PipelineSettings.
"""

import copy
import json
import random
from typing import Callable

import pytweening as easing

from modules.comfy.settings import PipelineSettings, Prefer

MAX_SEED: int = 2 ** 32 - 1     # the smallest maximum of the seeded nodes (QwenVL: 1 .. 4294967295)
FIXED_SEED: int = 1             # with portrait.fixed_seed: the description and the grin too, or the runs still differ

# LoraSettings field: title of its node
LORAS: dict[str, str] = {'clothing': 'LoRA: clothing', 'looking': 'LoRA: looking at viewer', 'dark': 'LoRA: dark'}

# inputs that name a model: by these names, or a file name with one of these extensions
MODEL_INPUTS: tuple[str, ...] = ('model_name', 'quantization', 'weight_dtype')
MODEL_EXTENSIONS: tuple[str, ...] = ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.onnx')
_SAME_CLASS: dict[str, str] = {'KSamplerFullPreview': 'KSampler'}   # build() swaps the sampler in

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
        # the nodes that take time, by the name the panel shows while they run
        self._steps: dict[str, str] = {self.describe: 'describe', self.depth_run: 'depth', self.sampler: 'portrait',
                                       self.grin: 'grin', self.last_depth_run: 'depth of last frame'}

    def step(self, node_id: str | None) -> str | None:
        """The step of the generation a node is, or None for the nodes in between."""
        return self._steps.get(node_id) if node_id is not None else None

    def build(self, image_name: str, style: str, negative: str, s: PipelineSettings) -> dict:
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

        for node_id in (self.sampler, self.describe):
            workflow[node_id]['inputs']['seed'] = self._seed(s)
        self._fill_grin(workflow, s)
        return workflow

    def build_replay(self, previous: dict, s: PipelineSettings) -> dict:
        """The previous workflow with only the grin filled in again (the grin loop). Everything up to
        the portrait is unchanged, so ComfyUI takes it from its cache and only makes the grin."""
        workflow: dict = copy.deepcopy(previous)
        self._fill_grin(workflow, s)
        return workflow

    def _fill_grin(self, workflow: dict, s: PipelineSettings) -> None:
        g = s.grin
        grin: dict = workflow[self.grin]['inputs']
        grin.update(frames=g.frames, ease=g.ease.value, crop_factor=g.face_region, midpoint=g.midpoint,
                    midpoint_deviation=g.midpoint_deviation, stitching=g.stitching)

        # a still has nothing random left, so its seed is fixed: unchanged settings give an identical workflow
        still: bool = g.still
        seed: int = FIXED_SEED if still else self._seed(s)
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
        grin.update(seed=seed, still=still)
        workflow[self.last_frame]['inputs']['batch_index'] = 0 if still else g.frames - 1

    def frame_count(self, workflow: dict) -> int:
        """Grin frames this workflow makes: one for a still."""
        grin: dict = workflow[self.grin]['inputs']
        return 1 if grin['still'] else grin['frames']

    def same_pipeline(self, prompt: dict | None) -> bool:
        """True when *prompt* (a workflow ComfyUI ran) is this pipeline: the same nodes, links and
        models, so ComfyUI still has everything loaded. The values filled in per run are not compared."""
        if not isinstance(prompt, dict) or prompt.keys() != self._template.keys():
            return False
        for node_id, node in self._template.items():
            other: dict = prompt[node_id]
            if _SAME_CLASS.get(other.get('class_type'), other.get('class_type')) != node['class_type']:
                return False
            for name, value in node['inputs'].items():
                model: bool = name in MODEL_INPUTS or (isinstance(value, str) and value.lower().endswith(MODEL_EXTENSIONS))
                if (model or isinstance(value, list)) and other.get('inputs', {}).get(name) != value:
                    return False
        return True

    @staticmethod
    def _seed(s: PipelineSettings) -> int:
        return FIXED_SEED if s.portrait.fixed_seed else random.randint(1, MAX_SEED)
