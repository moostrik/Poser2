"""Each pass of the 3D motion and the 3D bridge alone, on saved frames: GL timer queries, the median of `--repeats`.

FRAMES is what depth_velocity_capture saved from one loop of a recording. The motion (`LucasKanadeFlow3D`, with the
frames' own settings) and a `Bridge3DLayer` at deep_flow's preset run through the first `--frame` frames, so their
memories hold real motion; then on that frame each part repeats: the motion whole, its x/y solve, its depth
velocity; the bridge's lowpass, its bandpass (with and without the accelerations) and the whole bridge. A repeat
measures the same pair of frames, so it costs what a camera frame costs.

    python scripts/motion_bridge_pass_check.py FRAMES [--frame 200] [--repeats 40]
"""

import ctypes
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import OpenGL
OpenGL.ERROR_CHECKING = False  # as launcher.py runs the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_TIME_ELAPSED, glBeginQuery, glEndQuery, glFinish, glGenQueries  # noqa: E402
from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v as raw_get_query  # noqa: E402

from modules.gl import Texture  # noqa: E402
from modules.flow import LucasKanadeFlow3D, LucasKanadeFlow3DSettings  # noqa: E402
from modules.render.layers import Bridge3DLayer, Bridge3DLayerSettings, BridgeInput  # noqa: E402
from bridge_compare import PRESET, depth_range  # noqa: E402
from depth_velocity_compare import Frames, context, recorded_settings, upload  # noqa: E402

GL_QUERY_RESULT = 0x8866


def median_ms(work, repeats: int, query: int) -> float:
    for _ in range(3):
        work()
    glFinish()
    times: list[float] = []
    for _ in range(repeats):
        glBeginQuery(GL_TIME_ELAPSED, query)
        work()
        glEndQuery(GL_TIME_ELAPSED)
        value = ctypes.c_uint64(0)
        raw_get_query(query, GL_QUERY_RESULT, ctypes.byref(value))
        times.append(value.value / 1e6)
    return float(np.median(times))


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path)
    parser.add_argument('--frame', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=40)
    args = parser.parse_args()
    frames = Frames(args.frames)
    context(False, 64, 64)
    query = glGenQueries(1)[0]

    flow_settings = LucasKanadeFlow3DSettings()
    flow_settings.update_from_dict(recorded_settings(frames.calls[0]['settings']))
    flow = LucasKanadeFlow3D(flow_settings)
    flow.allocate(frames.width, frames.height)
    bridge_settings = Bridge3DLayerSettings()
    bridge_settings.update_from_dict(json.loads(PRESET.read_text())['render']['bridge'])
    bridge = Bridge3DLayer(bridge_settings)
    bridge.allocate(2160, 3840)

    luminance, depth, mask = Texture(), Texture(), Texture()
    last = min(args.frame, len(frames) - 1)
    for k in range(last + 1):
        lum, dep, msk = frames.load(k)
        upload(luminance, lum)
        upload(depth, dep)
        upload(mask, msk.astype(np.float32))
        c = frames.calls[k]
        flow.set_input(luminance)
        flow.set_mask(mask)
        flow.update(depth, c['delta_time'], c['view_span'], tuple(c['depth_step']), depth_range=depth_range())
        bridge.set_input(BridgeInput(flow.velocity, c['delta_time'], None, mask=mask, z_position=flow.z_position))
        bridge.update()
    glFinish()
    c = frames.calls[last]
    z = flow.settings.z
    xy = flow.settings.xy
    dt, span, step = c['delta_time'], c['view_span'], tuple(c['depth_step'])
    dv = flow._depth_velocity

    rows = [
        ('motion, whole', lambda: flow.update(depth, dt, span, step, depth_range=depth_range())),
        ('motion: x/y solve (pyramid, threshold)', lambda: flow._flow.update(depth, z.tolerance)),
        ('motion: depth velocity', lambda: dv.update_depth(depth, dt, span, step, z.window, z.smooth, z.threshold, z.tolerance,
                                                           z.dead_zone, z.catch_up, xy.threshold, flow._flow.solved, z.edge,
                                                           z.max_speed, None)),
        ('bridge: lowpass', lambda: bridge._lowpass.update(dt, False)),
        ('bridge: lowpass with acceleration', lambda: bridge._lowpass.update(dt, True)),
        ('bridge: bandpass', lambda: bridge._bandpass.update(dt, False)),
        ('bridge: bandpass with accelerations', lambda: bridge._bandpass.update(dt, True)),
        ('bridge, whole', lambda: bridge.update()),
    ]
    print(f'{frames.width} x {frames.height}, frame {last}; levels {xy.levels}, window {int(xy.window)}')
    for name, work in rows:
        print(f'{name:<42} {median_ms(work, args.repeats, query):7.3f} ms', flush=True)


if __name__ == '__main__':
    main()
