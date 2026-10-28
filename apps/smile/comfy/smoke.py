"""Smoke test of the ComfyUI install: the last step of scripts\\install.bat, run with the app venv.

Starts ComfyUI from its own venv the way start_comfy.bat does, waits for it, checks that it runs on CUDA
and that onnxruntime has its CUDA provider, runs the warm-up workflow once (every model loads, the whole
chain works), then stops it. A failure names the step. Exit 0 when all passed.

    .venv\\Scripts\\python -m apps.smile.comfy.smoke [--comfy DIR] [--port 8188] [--preset FILE] [--keep] [--save DIR]

--save runs the whole workflow with the fixed seed instead and writes its portrait and last grin frame into
DIR: the picture of an install, which install_comfy.ps1 -Update makes before and after to put side by side.
Refuses when the port is taken: a ComfyUI that is already running (the dev install on this PC, SMILE's own)
would be tested instead of the install.
"""

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from apps.smile.comfy.client import ComfyClient, Event
from apps.smile.comfy.Generator import WARMUP_IMAGE, WARMUP_TIMEOUT
from apps.smile.comfy.workflow import NO_LORA, Workflow
from apps.smile.settings import Settings
from modules.settings import presets

logger = logging.getLogger('smoke')

ROOT: Path = Path(__file__).resolve().parents[2]
START_TIMEOUT: float = 240.0        # s for ComfyUI to answer /system_stats (the packs import torch, insightface, transformers)
PRESET: str = 'apps/smile/data/settings/Smile.json'


class Failed(Exception):
    pass


def port_taken(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def check_onnxruntime(python: Path) -> None:
    result = subprocess.run([str(python), '-c', 'import onnxruntime as o; print(o.get_available_providers())'],
                            capture_output=True, text=True)
    if 'CUDAExecutionProvider' not in result.stdout:
        raise Failed(f'onnxruntime in the ComfyUI venv has no CUDA provider (face analysis would run on the CPU): {result.stdout.strip() or result.stderr.strip()}')
    logger.info('onnxruntime CUDA provider: ok')


def start_comfy(comfy: Path, python: Path, port: int, log: Path) -> subprocess.Popen:
    env: dict[str, str] = dict(os.environ, HF_HUB_OFFLINE='1', YOLO_VERBOSE='false')
    command: list[str] = [str(python), 'main.py', '--preview-method', 'auto', '--port', str(port), '--verbose', 'INFO', str(log)]
    logger.info('starting ComfyUI: %s', ' '.join(command[1:]))
    return subprocess.Popen(command, cwd=comfy, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for(client: ComfyClient, process: subprocess.Popen, timeout: float) -> None:
    deadline: float = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise Failed(f'ComfyUI ended with exit code {process.returncode} before it answered, see its log')
        try:
            client.connect()
            return
        except Exception:
            time.sleep(2.0)
    raise Failed(f'ComfyUI did not answer within {timeout:.0f} s, see its log')


def check_cuda(log: Path) -> None:
    text: str = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    line: str | None = next((l for l in text.splitlines() if 'Device:' in l), None)
    if line is None or 'cuda' not in line:
        raise Failed(f'ComfyUI does not run on CUDA: {line or "no Device line in its log"}')
    logger.info('%s', line.strip())


def warm_up(client: ComfyClient, preset: str, save: Path | None = None) -> float:
    """The warm-up workflow once, as the generator runs it after every connect. The seconds it took.
    With save: the full run instead, with the fixed seed, and its portrait and last grin frame written
    into that folder as PNGs (the picture of an install, to put next to another's)."""
    settings = Settings()
    if not presets.load(settings, preset):
        raise Failed(f'could not load the preset {preset}')
    p = settings.pipeline
    sampler: dict = client.object_info('KSampler')['input']['required']
    loras: list[str] = list(client.object_info('LoraLoader')['input']['required']['lora_name'][0])
    p.portrait.sampler_options = list(sampler['sampler_name'][0])
    p.portrait.scheduler_options = list(sampler['scheduler'][0])
    p.portrait.checkpoint_options = list(client.object_info('CheckpointLoaderSimple')['input']['required']['ckpt_name'][0])
    for style in (p.portrait, p.style_a, p.style_b):
        style.lora_options = [NO_LORA] + loras
    for name in (p.portrait.checkpoint, p.portrait.distill_lora, p.style_a.lora_1, p.style_a.lora_2, p.style_a.lora_3):
        if name != NO_LORA and name not in p.portrait.checkpoint_options + p.portrait.lora_options:
            raise Failed(f'ComfyUI does not offer {name}: the models folder misses it (run the model tool)')

    image = cv2.imread(str(ROOT / WARMUP_IMAGE))
    if image is None:
        raise Failed(f'no warm-up image at {WARMUP_IMAGE}')
    wf = Workflow(str(ROOT / p.connection.workflow))
    name: str = client.upload(image, 'smile_smoke.png')
    if save is not None:
        p.portrait.fixed_seed = True
        workflow: dict = wf.build(name, p.style_a, p)
    else:
        workflow = wf.build_warmup(name, p.style_a, p)
    started: float = time.monotonic()
    try:
        prompt_id: str = client.queue(workflow)
    except RuntimeError as e:
        raise Failed(f'{e} (a node of the workflow changed its inputs? re-export data/comfy/smile_workflow.json from the ComfyUI page)') from e
    logger.info('%s queued (%d steps)', 'fixed-seed run' if save else 'warm-up', wf.step_count)
    deadline: float = started + WARMUP_TIMEOUT
    last_grin_frame = None
    portrait = None
    while time.monotonic() < deadline:
        event: Event | None = client.receive(5.0)
        if event is None:
            continue
        if event.type == 'preview' and event.image is not None and event.data.get('node_id') == wf.grin:
            last_grin_frame = event.image
            continue
        if event.data.get('prompt_id') not in (None, prompt_id):
            continue
        if event.type == 'executing' and event.data.get('node'):
            step: str | None = wf.step(event.data['node'])
            if step:
                logger.info('  %s', step)
        elif event.type == 'executed' and event.data.get('node') == wf.portrait:
            images: list[dict] = event.data.get('output', {}).get('images', [])
            portrait = client.view(images[0]) if images else None
        elif event.type == 'execution_success' or (event.type == 'executing' and event.data.get('node') is None and 'prompt_id' in event.data):
            seconds: float = time.monotonic() - started
            if save is not None:
                save.mkdir(parents=True, exist_ok=True)
                if portrait is not None:
                    cv2.imwrite(str(save / 'portrait.png'), portrait)
                if last_grin_frame is not None:
                    cv2.imwrite(str(save / 'grin_last.png'), last_grin_frame)
                logger.info('saved the portrait and the last grin frame into %s', save)
            return seconds
        elif event.type in ('execution_error', 'execution_interrupted'):
            raise Failed(f"the run failed in ComfyUI: {event.data.get('exception_message') or event.type} (node {event.data.get('node_type')})")
    raise Failed(f'the run did not finish within {WARMUP_TIMEOUT:.0f} s')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--comfy', type=Path, default=ROOT / 'ComfyUI', help='the ComfyUI install')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8188)
    parser.add_argument('--preset', default=PRESET, help='the preset whose models and workflow are tested')
    parser.add_argument('--keep', action='store_true', help='leave ComfyUI running afterwards')
    parser.add_argument('--save', type=Path, help='run the full workflow with the fixed seed and save its portrait and last grin frame here')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', stream=sys.stderr)

    python: Path = args.comfy / '.venv' / 'Scripts' / 'python.exe'
    log_dir: Path = ROOT / 'output' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log: Path = log_dir / f'comfy_smoke_{datetime.now():%Y%m%d_%H%M%S}.log'
    process: subprocess.Popen | None = None
    client = ComfyClient(args.host, args.port)
    try:
        if not python.exists():
            raise Failed(f'no ComfyUI venv at {python}: run scripts\\install_comfy.ps1')
        if port_taken(args.host, args.port):
            raise Failed(f'port {args.port} is in use: close the running ComfyUI (startSMILE.bat, or the dev install) and run again')
        check_onnxruntime(python)
        process = start_comfy(args.comfy, python, args.port, log)
        wait_for(client, process, START_TIMEOUT)
        check_cuda(log)
        seconds: float = warm_up(client, args.preset, args.save)
        print(f'Smoke test passed: ComfyUI on CUDA, the {"fixed-seed run" if args.save else "warm-up"} took {seconds:.0f} s (log {log})')
        return 0
    except Failed as e:
        print(f'Smoke test FAILED: {e}' + (f' (log {log})' if log.exists() else ''))
        return 1
    finally:
        client.close()
        if process is not None and not args.keep:
            # the whole tree: ComfyUI may have children of its own that would keep its folder in use
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, creationflags=0x08000000)
            try:
                process.wait(15.0)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == '__main__':
    sys.exit(main())
