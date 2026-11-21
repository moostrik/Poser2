"""Save what the Lucas–Kanade flow gets, frame by frame, from one loop of a recording, for depth_velocity_compare.

    python scripts/depth_velocity_capture.py RECORDING [--out temp/depth_velocity] [--name NAME] [--seconds S]
                                             [--set PATH=VALUE ...]

Runs deep_flow on the recording (as -sim, without the panel), from the start of its loop to the next, or
for its first `--seconds`, with the settings file's settings and the Lucas–Kanade flow. Each measure of the flow is
saved to OUT/NAME (the recording's name without `--name`): `luminance_NNNN.npy` (the newest luminance frame,
0 to 1), `depth_NNNN.npy` (the depth it got, m, at the flow's size, before the step filter), `confidence_NNNN.npy`
(the depth's confidence, 0 to 1, 0 where the view filled the depth), `mask_NNNN.npy`
(the player's mask, 0 to 1, the weight of each pixel), and `calls.json`: each
measure's arguments and its moment, `t`, in seconds, and the flow's settings; and `capture.json`: the recording,
the commit, `--set` and the view's settings it ran with. The compare replays them from
no motion. About 3 GB for 15 seconds at the flow's 448×800: delete it when done with it. This is the one
step that runs the app.
"""

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import cast

os.environ.setdefault('DEPTHAI_LEVEL', 'error')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OpenGL  # noqa: E402
OpenGL.ERROR_CHECKING = False
import numpy as np  # noqa: E402
from OpenGL.GL import GL_FLOAT, GL_PACK_ALIGNMENT, GL_RGBA, GL_RGBA32F, GL_TEXTURE_2D, glBindTexture, glPixelStorei  # noqa: E402
from OpenGL.raw.GL.VERSION.GL_1_1 import glGetTexImage as raw_get_tex_image  # noqa: E402

from modules.log_config import setup_logging, install_excepthooks  # noqa: E402
from modules.gl import Fbo, Style, Texture  # noqa: E402
from modules.flow import FlowUtil, LucasKanadeFlow3D  # noqa: E402
from modules import render  # noqa: E402
from apps.deep_flow.main import DeepFlowMain  # noqa: E402

LOOP_TIMEOUT: float = 60.0      # seconds a loop may take before the capture gives up
SHORTEST_LOOP: float = 5.0      # seconds a loop takes at least: a loop call back sooner after its start is the start's


def parsed(value: str, current: object) -> object:
    """`value` as the type of the setting it replaces: an enum by its name or number, a bool by its word."""
    if isinstance(current, Enum):
        kind = type(current)
        return kind[value.upper()] if not value.lstrip('-').isdigit() else kind(int(value))
    if isinstance(current, bool):
        return value.lower() in ('1', 'true', 'on', 'yes')
    return cast(type[float], type(current))(float(value))     # a number: int or float


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recording', help='the recording folder name under the app\'s recordings')
    parser.add_argument('--out', type=Path, default=Path('temp/depth_velocity'))
    parser.add_argument('--name', default='', help='the capture\'s folder under OUT (default: the recording\'s name)')
    parser.add_argument('--seconds', type=float, default=0.0, help='stop this long after the loop starts (default: the whole loop)')
    parser.add_argument('--set', action='append', default=[], metavar='PATH=VALUE',
                        help="an app setting by its path for this capture, e.g. flow.view.mask.guided_radius=3")
    args = parser.parse_args()
    out_dir: Path = args.out / (args.name or args.recording)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(verbose=False, name='deep_flow')
    install_excepthooks()
    targets: dict[str, Fbo] = {}
    calls: list[dict] = []
    recording, starting, done = threading.Event(), threading.Event(), threading.Event()

    def read(name: str, source, w: int, h: int) -> np.ndarray:
        if name not in targets:
            targets[name] = Fbo()
            targets[name].allocate(w, h, GL_RGBA32F)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        FlowUtil.blit(targets[name], source)
        Style.pop_style()
        array = np.empty((h, w, 4), np.float32)
        glBindTexture(GL_TEXTURE_2D, targets[name].tex_id)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        raw_get_tex_image(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, array.ctypes.data_as(ctypes.c_void_p))
        glBindTexture(GL_TEXTURE_2D, 0)
        return array

    update = LucasKanadeFlow3D.update
    held: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []     # the frames, saved after the loop: saving in it costs captures
    gaps: list[float | None] = []                                   # each measure's gap between two capture times, as it is
    frame_interval = render.frame_interval

    def gap_kept(timestamp, last, *rest) -> float:
        gaps.append(timestamp - last if timestamp is not None and last is not None else None)
        return frame_interval(timestamp, last, *rest)

    render.frame_interval = gap_kept

    def saved(self: LucasKanadeFlow3D, depth: Texture | None, delta_time: float, view_span: float,
              depth_step: tuple[float, float], guarded: bool = False,
              depth_range: tuple[float, float] = (0.0, 0.0)) -> None:
        if starting.is_set():                     # on the main thread: the loop starts here
            starting.clear()
            recording.set()
        if recording.is_set() and depth is not None:
            luminance = self.luminance
            w, h = luminance.width, luminance.height
            mask = self._mask if self._mask is not None and self._mask.allocated else None
            depth_read = read('depth', depth, w, h)
            held.append((read('luminance', luminance, w, h)[..., 0].copy(), depth_read[..., 0].copy(),
                         depth_read[..., 1].astype(np.float16),
                         read('mask', mask, w, h)[..., 0].astype(np.float16) if mask is not None else np.ones((h, w), np.float16)))
            calls.append(dict(t=time.perf_counter(), delta_time=delta_time, gap=gaps[-1] if gaps else None,
                              view_span=view_span, depth_step=list(depth_step),
                              depth_range=list(depth_range), width=w, height=h,
                              settings=self.settings.to_dict()))
            if args.seconds > 0.0 and calls[-1]['t'] - calls[0]['t'] >= args.seconds:
                recording.clear()
                done.set()
        update(self, depth, delta_time, view_span, depth_step, guarded=guarded, depth_range=depth_range)

    LucasKanadeFlow3D.update = saved
    app = DeepFlowMain(simulation=True, interface=False)
    loops = threading.Semaphore(0)
    assert app.player is not None, 'deep_flow made no player: the recording needs -sim'
    app.player.add_loop_callback(loops.release)

    def run() -> None:
        try:
            time.sleep(8.0)                       # the app and its models come up
            for item in args.set:                 # the settings the capture runs with, beside the settings file's
                path, value = item.split('=', 1)
                *groups, name = path.split('.')
                owner = app.settings
                for group in groups:
                    owner = getattr(owner, group)
                setattr(owner, name, parsed(value, getattr(owner, name)))
            simulator = app.settings.setup.camera.simulator
            simulator.folder = args.recording
            time.sleep(0.5)
            simulator.start = True
            loops.acquire(timeout=LOOP_TIMEOUT)   # the loop's start
            starting.set()
            time.sleep(0.5)
            while loops.acquire(blocking=False):  # its start can call back twice
                pass
            began: float = time.perf_counter()
            deadline: float = began + LOOP_TIMEOUT
            while not done.is_set() and time.perf_counter() < deadline:     # the next loop, or --seconds of it
                if loops.acquire(timeout=0.1) and time.perf_counter() - began >= SHORTEST_LOOP:
                    break                         # a call back sooner is its start's again, not the next loop
            recording.clear()
            time.sleep(0.2)                       # the main thread's last measure is in
            for index, (luminance, depth, confidence, mask) in enumerate(held[:len(calls)]):
                np.save(out_dir / f'luminance_{index:04d}.npy', luminance)
                np.save(out_dir / f'depth_{index:04d}.npy', depth)
                np.save(out_dir / f'confidence_{index:04d}.npy', confidence)
                np.save(out_dir / f'mask_{index:04d}.npy', mask)
            del calls[len(held):]
            (out_dir / 'calls.json').write_text(json.dumps(calls, default=str))
            root = Path(__file__).resolve().parent.parent
            head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=root).stdout.strip()
            (out_dir / 'capture.json').write_text(json.dumps(dict(
                recording=args.recording, commit=head, date=time.strftime('%Y-%m-%d %H:%M'), seconds=args.seconds,
                set=args.set, view=app.settings.flow.view.to_dict()), default=str))
            print(f'{len(calls)} frames to {out_dir}', flush=True)
        finally:
            app.stop()                            # whatever happened, the app does not keep running

    app.start()
    threading.Thread(target=run, daemon=True).start()
    app.render.run()                  # the window loop on the main thread, until the probe stops the app
    while not app.is_finished:
        time.sleep(0.05)
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
