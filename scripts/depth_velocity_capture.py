"""Save what the Lucas–Kanade flow gets, frame by frame, from one loop of a recording, for depth_velocity_compare.

    python scripts/depth_velocity_capture.py RECORDING [--out temp/depth_velocity] [--set PATH=VALUE ...]

Runs deep_flow on the recording (as -sim, without the panel), from the start of its loop to the next,
with the preset's settings and the Lucas–Kanade flow. Each measure of the flow is saved to
OUT/RECORDING/: `luminance_NNNN.npy` (the newest luminance frame, 0 to 1), `depth_NNNN.npy` (the
depth it got, m, at the flow's size, before the step filter), `mask_NNNN.npy` (the player's mask),
and `calls.json`: each measure's arguments and its moment, `t`, in seconds, and the flow's settings.
The compare replays them from no motion. About 3 GB for
15 seconds at the flow's 448×800: delete it when done with it. This is the one step that runs the app.
"""

import ctypes
import json
import os
import sys
import threading
import time
from argparse import ArgumentParser
from pathlib import Path

os.environ.setdefault('DEPTHAI_LEVEL', 'error')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OpenGL  # noqa: E402
OpenGL.ERROR_CHECKING = False
import numpy as np  # noqa: E402
from OpenGL.GL import GL_FLOAT, GL_PACK_ALIGNMENT, GL_RGBA, GL_RGBA32F, GL_TEXTURE_2D, glBindTexture, glPixelStorei  # noqa: E402
from OpenGL.raw.GL.VERSION.GL_1_1 import glGetTexImage as raw_get_tex_image  # noqa: E402

from modules.log_config import setup_logging, install_excepthooks  # noqa: E402
from modules.gl import Fbo, Style  # noqa: E402
from modules.flow import FlowUtil, LucasKanadeFlow3D  # noqa: E402
from apps.deep_flow.main import DeepFlowMain  # noqa: E402

LOOP_TIMEOUT: float = 60.0      # seconds a loop may take before the capture gives up


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recording', help='the recording folder name under the app\'s recordings')
    parser.add_argument('--out', type=Path, default=Path('temp/depth_velocity'))
    parser.add_argument('--set', action='append', default=[], metavar='PATH=VALUE',
                        help="an app setting by its path for this capture, e.g. render.center.mask.guided_radius=0.02")
    args = parser.parse_args()
    out_dir: Path = args.out / args.recording
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(verbose=False, name='deep_flow')
    install_excepthooks()
    targets: dict[str, Fbo] = {}
    calls: list[dict] = []
    recording, starting = threading.Event(), threading.Event()

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

    def saved(self, depth, delta_time, view_span, depth_step, **flags) -> None:
        if starting.is_set():                     # on the render thread: the loop starts here
            starting.clear()
            recording.set()
        if recording.is_set() and depth is not None:
            luminance = self.luminance
            w, h = luminance.width, luminance.height
            index = len(calls)
            mask = self._mask if self._mask is not None and self._mask.allocated else None
            np.save(out_dir / f'luminance_{index:04d}.npy', read('luminance', luminance, w, h)[..., 0])
            np.save(out_dir / f'depth_{index:04d}.npy', read('depth', depth, w, h)[..., 0])
            np.save(out_dir / f'mask_{index:04d}.npy', (read('mask', mask, w, h)[..., 0] > 0.5) if mask is not None else np.ones((h, w), bool))
            calls.append(dict(t=time.perf_counter(), delta_time=delta_time, view_span=view_span, depth_step=list(depth_step),
                              width=w, height=h, settings=self.settings.to_dict()))
        update(self, depth, delta_time, view_span, depth_step, **flags)

    LucasKanadeFlow3D.update = saved
    app = DeepFlowMain(simulation=True, interface=False)
    loops = threading.Semaphore(0)
    assert app.player is not None, 'deep_flow made no player: the recording needs -sim'
    app.player.add_loop_callback(loops.release)

    def run() -> None:
        time.sleep(8.0)                           # the app and its models come up
        for item in args.set:                     # the settings the capture runs with, beside the preset's
            path, value = item.split('=', 1)
            *groups, name = path.split('.')
            owner = app.settings
            for group in groups:
                owner = getattr(owner, group)
            setattr(owner, name, type(getattr(owner, name))(float(value)))
        simulator = app.settings.camera.simulator
        simulator.folder = args.recording
        time.sleep(0.5)
        simulator.start = True
        loops.acquire(timeout=LOOP_TIMEOUT)       # the loop's start
        starting.set()
        time.sleep(0.5)
        while loops.acquire(blocking=False):      # its start can call back twice
            pass
        loops.acquire(timeout=LOOP_TIMEOUT)       # the next loop: one loop recorded
        recording.clear()
        (out_dir / 'calls.json').write_text(json.dumps(calls, default=str))
        print(f'{len(calls)} frames to {out_dir}', flush=True)
        app.stop()

    app.start()
    threading.Thread(target=run, daemon=True).start()
    while not app.is_finished:
        time.sleep(0.05)
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
