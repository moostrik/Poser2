"""A test workflow for the ComfyUI page: what SMILE sends for one visitor, with a preset's values,
plus previews and a video recorder. Generated, so it can't drift from the app:

    .venv\\Scripts\\python -m apps.smile.comfy.testflow [-p preset] [-s a|b] [-o file]

The input is the app's own capture (SMILE uploads every capture as smile_capture.png); for any other
photo, crop it first as SMILE would (python -m apps.smile.face.crop photo.jpg) and pick the crop in the
Load Image node. API format: drag the file into the page, which lays the nodes out itself.
"""

import argparse
import copy
import json
import os

from apps.smile.comfy.workflow import Workflow, _find
from modules.settings import presets
from apps.smile.settings import Settings

CAPTURE: str = 'smile_capture.png'      # the name Generator uploads every capture under
PROMPT: str = 'test_prompt'
VIDEO: str = 'test_video'
VIDEO_FILE: str = 'test_video_file'


def make_test_workflow(workflow_path: str, settings: Settings, style: str = 'a') -> dict:
    wf: Workflow = Workflow(workflow_path)
    p = settings.pipeline
    workflow: dict = wf.build(CAPTURE, getattr(p, f'style_{style}'), p)
    with open(workflow_path, encoding='utf-8') as f:
        source: dict = json.load(f)     # for the nodes Workflow strips: they keep the file's settings

    def title(node_id: str, text: str) -> None:
        workflow[node_id]['_meta'] = {'title': text}

    title(wf.load_image, "Input (the app's capture, or a photo cropped with apps.smile.face.crop)")
    workflow[PROMPT] = {'class_type': 'PreviewAny', 'inputs': {'source': [wf.suffix, 0]}, '_meta': {'title': 'Prompt'}}
    title(wf.description, 'Description')
    title(wf.capture_depth, 'Depth of capture')
    title(wf.last_depth, 'Depth of last frame')
    title(wf.prefix, 'Prompt prefix')
    title(wf.last_frame, 'Last grin frame')
    title(wf.last_depth_run, 'Depth of last frame (run)')

    # from the workflow file, as in the old tests: capture over portrait; the grin as a video (ComfyUI's own nodes)
    overlay: str = _find(source, title='Overlay')
    for node_id in (overlay, _find(source, title='Overlay preview')):
        workflow[node_id] = copy.deepcopy(source[node_id])
    workflow[VIDEO] = {'class_type': 'CreateVideo', 'inputs': {'images': [wf.grin, 0], 'fps': settings.sequence.grin_fps},
                       '_meta': {'title': 'Grin video'}}
    workflow[VIDEO_FILE] = {'class_type': 'SaveVideo', 'inputs': {'video': [VIDEO, 0], 'filename_prefix': 'smile_test',
                                                                  'format': 'mp4', 'codec': 'h264'},
                            '_meta': {'title': 'Save grin video'}}
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(description='Write the test workflow for the ComfyUI page')
    parser.add_argument('-p', '--preset', default='apps/smile/data/settings/Smile.json', help='preset with the values to fill in')
    parser.add_argument('-s', '--style', default='a', choices=['a', 'b'], help='which of the two styles')
    parser.add_argument('-o', '--out', default='apps/smile/data/comfy/smile_test.json', help='file to write')
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
