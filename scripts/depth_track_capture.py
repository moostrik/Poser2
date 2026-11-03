"""Save what the depth track gets, frame by frame, from one loop of a recording, for depth_track_compare.

    python scripts/depth_track_capture.py RECORDING [--out temp/depth_track]

Runs deep_flow on the recording (as -sim, without the panel), from the start of its loop to the next,
with the preset's settings. Each call into the Lucas–Kanade flow's depth track is saved to
OUT/RECORDING/: `depth_NNNN.npy` (the depth it got, m, at the flow's size, before the step filter),
`moved_NNNN.npy` (the displacement it is carried by, uv per frame), `mask_NNNN.npy` (the player's
mask), and `calls.json` (each call's arguments and its moment, `t`, in seconds). The track starts
over at the loop's start, so the saved calls follow from no track. About 2.4 GB for 15 seconds at
the flow's 448×800: delete it when done with it. This is the one step that runs the app.
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
from modules.flow import DepthVelocity, FlowUtil  # noqa: E402
from modules.render.layers.flow.flow_layer import OpticalFlowMethod  # noqa: E402
from apps.deep_flow.main import DeepFlowMain  # noqa: E402
from apps.deep_flow.settings import Layers  # noqa: E402

LOOP_TIMEOUT: float = 60.0      # seconds a loop may take before the capture gives up


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recording', help='the recording folder name under the app\'s recordings')
    parser.add_argument('--out', type=Path, default=Path('temp/depth_track'))
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

    update_depth = DepthVelocity.update_depth

    def saved(self, depth, delta_time, view_span, depth_step, window, time_, threshold, tolerance, dead_zone, catch_up_time,
              motion_floor, displacement=None):
        if starting.is_set():                     # on the render thread: the loop starts from no track
            starting.clear()
            self.reset()
            recording.set()
        if recording.is_set() and displacement is not None:
            w, h = self._input_fbo.width, self._input_fbo.height
            index = len(calls)
            layer = app.render.L[Layers.flow].get(0)
            mask = layer._mask_input if layer is not None and layer._mask_input is not None and layer._mask_input.allocated else None
            np.save(out_dir / f'depth_{index:04d}.npy', read('depth', depth, w, h)[..., 0])
            np.save(out_dir / f'moved_{index:04d}.npy', read('moved', displacement, w, h)[..., :2])
            np.save(out_dir / f'mask_{index:04d}.npy', (read('mask', mask, w, h)[..., 0] > 0.5) if mask is not None else np.ones((h, w), bool))
            calls.append(dict(t=time.perf_counter(), delta_time=delta_time, view_span=view_span, depth_step=list(depth_step), window=window,
                              time=time_, threshold=threshold, tolerance=tolerance, dead_zone=dead_zone, catch_up_time=catch_up_time,
                              motion_floor=motion_floor, width=w, height=h))
        update_depth(self, depth, delta_time, view_span, depth_step, window, time_, threshold, tolerance, dead_zone, catch_up_time,
                     motion_floor, displacement)

    DepthVelocity.update_depth = saved
    app = DeepFlowMain(simulation=True, interface=False)
    app.settings.render.flow.optical_flow.method = OpticalFlowMethod.LUCAS_KANADE
    loops = threading.Semaphore(0)
    assert app.player is not None, 'deep_flow made no player: the recording needs -sim'
    app.player.add_loop_callback(loops.release)

    def run() -> None:
        time.sleep(8.0)                           # the app and its models come up
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
        (out_dir / 'calls.json').write_text(json.dumps(calls))
        print(f'{len(calls)} frames to {out_dir}', flush=True)
        app.stop()

    app.start()
    threading.Thread(target=run, daemon=True).start()
    while not app.is_finished:
        time.sleep(0.05)
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
