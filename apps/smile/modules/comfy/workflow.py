"""The ComfyUI workflow (API format), adapted for SMILE and filled in per run.

SMILE crops and rotates the face itself (modules/face/FaceFinder.py), so the workflow's own
face crop is removed and its consumers read the uploaded image directly. The video and the
overlay check are removed too, and the saved portrait becomes a preview (nothing piles up in
ComfyUI's output folder). Nodes are found by class type or title, not by id.
"""

import copy
import json
import random

MAX_SEED: int = 2 ** 32 - 1     # the smallest maximum of the seeded nodes (QwenVL: 1 .. 4294967295)


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

        # the sampler previews every step with the real VAE (comfy/aiia_grin, KSamplerFullPreview)
        # instead of ComfyUI's approximate latent previews; the result is the same
        self.sampler: str = _find(workflow, class_type='KSampler')
        workflow[self.sampler]['class_type'] = 'KSamplerFullPreview'
        workflow[self.sampler]['inputs']['vae'] = list(workflow[_find(workflow, class_type='VAEDecode')]['inputs']['vae'])
        self.describe: str = _find(workflow, class_type='AILab_QwenVL')
        self.description: str = _find(workflow, title='Preview as Text')
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

        # the grin frames are the centre of the portrait: (x, y, w, h) relative to the portrait
        latent: dict = workflow[_find(workflow, class_type='EmptyLatentImage')]['inputs']
        crop: dict = workflow[_find(workflow, class_type='ImageCrop')]['inputs']
        self.portrait_size: tuple[int, int] = (latent['width'], latent['height'])
        self.grin_crop: tuple[float, float, float, float] = (crop['x'] / latent['width'], crop['y'] / latent['height'],
                                                             crop['width'] / latent['width'], crop['height'] / latent['height'])
        self._template: dict = workflow

    def build(self, image_name: str, style: str, frames: int) -> dict:
        workflow: dict = copy.deepcopy(self._template)
        workflow[self.load_image]['inputs']['image'] = image_name
        workflow[self.style]['inputs']['text'] = style
        workflow[self.grin]['inputs']['frames'] = frames
        workflow[self.last_frame]['inputs']['batch_index'] = frames - 1
        for node_id in (self.sampler, self.describe, self.grin):
            workflow[node_id]['inputs']['seed'] = random.randint(1, MAX_SEED)
        return workflow


def _midas(image: list) -> dict:
    return {'class_type': 'MiDaS-DepthMapPreprocessor',
            'inputs': {'image': image, 'a': 6.283185307179586, 'bg_threshold': 0.1, 'resolution': 1024}}
