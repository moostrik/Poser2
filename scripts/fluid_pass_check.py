"""Each pass of the 3D fluid alone against its floor: where a pass does work beyond moving its data.

The fluid at deep_flow's preset (`maccormack_check.fluid_settings`) runs `--warmup` seconds with the swinging body,
so its fields hold real values; then each pass is repeated `--repeats` times between GL timer queries and its
median kept. A pass's floor is the time to read each volume it needs once and write its result once at the GPU's
memory speed (`--bandwidth`, GB/s; the RTX 4090's 1008): a pass far above it spends its time on something else
(setup each cell repeats, interpolations, branches), one near it is bound by its memory. The density passes
count the upscaled amount too where the preset runs it. A pass that skips what it need not touch can run under
its floor; the floor counts every texel. Passes off at the preset
run with what turns them on (the viscosity, the vorticity, the buoyancy, MacCormack). Settings by path as in
maccormack_check: `--set velocity.confinement_1.strength=1`.

    python scripts/fluid_pass_check.py [--warmup 2] [--repeats 40] [--set path=value ...]
"""

import ctypes

import OpenGL
OpenGL.ERROR_CHECKING = False  # as launcher.py runs the app; a glGetError per call otherwise makes the CPU the bottleneck
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_TIME_ELAPSED, glBeginQuery, glEndQuery, glFinish, glGenQueries  # noqa: E402
from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v as raw_get_query  # noqa: E402

import monitor_harness as mh  # noqa: E402
import maccormack_check as mc  # noqa: E402
from modules.flow import Advection, Fluid3D  # noqa: E402

GL_QUERY_RESULT = 0x8866
FPS = 60.0


def median_ms(work: Callable[[], None], repeats: int, query: int) -> float:
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


def fluid_at(overrides: list[str], warmup: float, advection: Advection = Advection.SEMI_LAGRANGIAN) -> Fluid3D:
    fluid = Fluid3D(mc.fluid_settings(advection, overrides))
    fluid.allocate(2160, 3840)
    fluid.update(1.0 / FPS)
    for i in range(int(warmup * FPS)):
        texture = mh.texture_of(mc.body(i / FPS, True))
        fluid.drag_velocity(texture)
        fluid.update(1.0 / FPS)
        texture.deallocate()
    glFinish()
    return fluid


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--warmup', type=float, default=2.0)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--bandwidth', type=float, default=1008.0, help='GB/s')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    mh.context()
    query = glGenQueries(1)[0]
    on = ['velocity.viscosity=0.0001', 'velocity.confinement_1.strength=1', 'temperature.buoyancy=1']
    fluid = fluid_at(on + args.set, args.warmup)
    w, h, d = fluid._simulation_width, fluid._simulation_height, fluid._depth
    cells = w * h * d
    faces = (w + 1) * (h + 1) * (d + 1)
    output = fluid._output_width * fluid._output_height
    upscaled = fluid.settings.upscaled
    fine = upscaled.width * upscaled.height * upscaled.depth if upscaled.enabled else 0     # the upscaled amount's texels (R16F)
    body = mh.texture_of(mc.body(0.4, True))
    dye = mh.texture_of(np.clip(np.abs(mc.body(0.4, True)), 0.0, 1.0))
    iterations = 20

    # (name, the work, the bytes its floor moves)
    passes: list[tuple[str, Callable[[], None], float]] = [
        ('advect velocity', fluid._advect_velocity, 16.0 * faces),
        ('viscosity, 20 iterations', fluid._apply_viscosity, 16.0 * faces * iterations / 5),
        ('vorticity: curl, force, add', fluid._confine_vorticity, 8.0 * faces + 8.0 * cells * 4 + 16.0 * faces),
        ('advect temperature', fluid._advect_temperature, (2.0 + 8.0 + 2.0) * cells),
        ('buoyancy', fluid._apply_buoyancy, (2.0 + 8.0 + 8.0) * cells + 8.0 * cells + 16.0 * faces),
        ('pressure', fluid._enforce_incompressibility, 0.0),
        ('faces to centers', fluid._faces_to_centers, 8.0 * faces + 8.0 * cells),
        ('density', fluid._advect_density, (8.0 + 8.0 + 8.0) * cells + 4.0 * fine),
        ('light', fluid._update_light, 8.0 * cells / 64),
        ('composite', fluid._composite_output, 8.0 * cells + 8.0 * output),
        ('inject: drag', lambda: fluid.drag_velocity(body), 16.0 * faces),
        ('inject: body force', lambda: fluid.add_acceleration(body), 16.0 * faces),
        ('inject: boundary density', lambda: fluid.add_boundary_density(dye), 16.0 * cells + 4.0 * fine),
        ('inject: body force density', lambda: fluid.add_body_force_density(dye), 16.0 * cells + 4.0 * fine),
    ]
    print(f'{w} x {h} x {d} cells, the upscaled amount {fine / 1e6:.0f} M texels; floor at {args.bandwidth:.0f} GB/s')
    print(f"{'pass':<30} {'ms':>7} {'floor ms':>9} {'x floor':>8}")
    for name, work, floor_bytes in passes:
        ms = median_ms(work, args.repeats, query)
        floor = floor_bytes / (args.bandwidth * 1e9) * 1e3
        ratio = f'{ms / floor:8.1f}' if floor > 0 else f"{'-':>8}"
        print(f'{name:<30} {ms:7.3f} {floor:9.3f} {ratio}', flush=True)
    fluid.deallocate()
    fluid = fluid_at(on + args.set, args.warmup, Advection.MACCORMACK)
    ms = median_ms(fluid._advect_velocity, args.repeats, query)
    floor = 40.0 * faces / (args.bandwidth * 1e9) * 1e3
    print(f"{'advect velocity, MacCormack':<30} {ms:7.3f} {floor:9.3f} {ms / floor:8.1f}")
    ms = median_ms(fluid._advect_density, args.repeats, query)
    floor = (48.0 * cells + 6.0 * fine) / (args.bandwidth * 1e9) * 1e3
    print(f"{'density, MacCormack':<30} {ms:7.3f} {floor:9.3f} {ms / floor:8.1f}")
    fluid.deallocate()


if __name__ == '__main__':
    main()
