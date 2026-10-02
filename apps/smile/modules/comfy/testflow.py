"""A test workflow for the ComfyUI page: what SMILE sends for one visitor, with a preset's values,
plus an input switch, previews and a video recorder. Generated, so it can't drift from the app:

    .venv\\Scripts\\python -m modules.comfy.testflow [-p preset] [-s a|b] [-o file]

The input switch is off for the app's own capture (SMILE uploads every capture as smile_capture.png)
and on for any photo, which then goes through the face crop of the workflow file. API format: drag
the file into the page, which lays the nodes out itself.
"""

import argparse
import copy
import json
import os

from modules.comfy.workflow import Workflow, _find
from modules.settings import presets
from modules.main_settings import Settings

CAPTURE: str = 'smile_capture.png'      # the name Generator uploads every capture under
SWITCH: str = 'test_input'
PROMPT: str = 'test_prompt'
OUTPUT_CLASSES: tuple[str, ...] = ('PreviewImage', 'PreviewAny', 'SaveImage')


def make_test_workflow(workflow_path: str, settings: Settings, style: str = 'a') -> dict:
    wf: Workflow = Workflow(workflow_path)
    p = settings.pipeline
    workflow: dict = wf.build(CAPTURE, getattr(p, f'style_{style}'), p)
    with open(workflow_path, encoding='utf-8') as f:
        source: dict = json.load(f)     # for the nodes Workflow strips: they keep the file's settings

    def title(node_id: str, text: str) -> None:
        workflow[node_id]['_meta'] = {'title': text}

    # the face crop chain without its previews: nothing forces it to run while the switch is off
    cropper: str = _find(source, class_type='LivePortraitCropper')
    prefix: str = cropper.split(':')[0] + ':'
    crop_nodes: set[str] = {node_id for node_id, node in source.items()
                            if node_id.startswith(prefix) and node['class_type'] not in OUTPUT_CLASSES}
    crop_out: str = next(value[0] for node_id, node in source.items() if not node_id.startswith(prefix)
                         for value in node['inputs'].values() if isinstance(value, list) and value[0] in crop_nodes)
    for node_id in crop_nodes:
        workflow[node_id] = copy.deepcopy(source[node_id])

    # everything that read the uploaded image reads the switch: off the image as it is, on the crop
    for node_id, node in workflow.items():
        if node_id in crop_nodes:
            continue
        for name, value in node['inputs'].items():
            if value == [wf.load_image, 0]:
                node['inputs'][name] = [SWITCH, 0]
    workflow[SWITCH] = {'class_type': 'ComfySwitchNode',
                        'inputs': {'switch': False, 'on_false': [wf.load_image, 0], 'on_true': [crop_out, 0]},
                        '_meta': {'title': 'Use face crop (on for an uncropped photo)'}}

    workflow[PROMPT] = {'class_type': 'PreviewAny', 'inputs': {'source': [wf.suffix, 0]}, '_meta': {'title': 'Prompt'}}
    title(wf.description, 'Description')
    title(wf.capture_depth, 'Depth of capture')
    title(wf.last_depth, 'Depth of last frame')
    title(wf.prefix, 'Prompt prefix')
    title(wf.last_frame, 'Last grin frame')
    title(wf.last_depth_run, 'Depth of last frame (run)')

    # from the workflow file, as in the old tests: capture over portrait, and the grin as a video
    overlay: str = _find(source, title='Overlay')
    video: str = _find(source, class_type='VHS_VideoCombine')
    for node_id in (overlay, _find(source, title='Overlay preview'), video):
        workflow[node_id] = copy.deepcopy(source[node_id])
    workflow[video]['inputs'].update(images=[wf.grin, 0], frame_rate=settings.sequence.grin_fps, filename_prefix='smile_test')
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(description='Write the test workflow for the ComfyUI page')
    parser.add_argument('-p', '--preset', default='data/settings/Smile.json', help='preset with the values to fill in')
    parser.add_argument('-s', '--style', default='a', choices=['a', 'b'], help='which of the two styles')
    parser.add_argument('-o', '--out', default='data/comfy/smile_test.json', help='file to write')
    args = parser.parse_args()

    settings = Settings()
    if not presets.load(settings, args.preset):
        raise SystemExit(f'could not load preset {args.preset}')
    workflow: dict = make_test_workflow(settings.pipeline.connection.workflow, settings, args.style)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    print(f'{args.out}: {len(workflow)} nodes, preset {args.preset}, style {args.style.upper()}')


if __name__ == '__main__':
    main()
