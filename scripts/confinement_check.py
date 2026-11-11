"""The vorticity confinements against each other: the swinging body for `--seconds`, each run's velocity at
the end, the largest difference between runs that should agree, and how much the second changes the water.

    1 alone         confinement 1 (ε, R), the others off
    3 alone         confinement 3 (ε, R), the others off: the same water
    split           all three at (ε / 4, ε / 4, ε / 2, R): the forces summed, the same water
    two sizes       confinement 1 (ε, R), confinement 2 (ε, r): what a second size adds

deep_flow's preset otherwise (`maccormack_check.fluid_settings`). No app: a hidden window gives the GL context.

    python scripts/confinement_check.py [--seconds 2] [--strength 2] [--radius 0.02] [--small 0.006]
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from OpenGL.GL import glFinish

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.flow import Fluid3D  # noqa: E402
from modules.flow.fluid_3d import Advection  # noqa: E402
import monitor_harness as mh  # noqa: E402
from maccormack_check import FPS, body, fluid_settings  # noqa: E402


def run(overrides: list[str], seconds: float) -> np.ndarray:
    fluid = Fluid3D(fluid_settings(Advection.SEMI_LAGRANGIAN, overrides))
    fluid.allocate(2160, 3840)
    for i in range(int(seconds * FPS)):
        texture = mh.texture_of(body(i / FPS, True))
        fluid.drag_velocity(texture)
        fluid.update(1.0 / FPS)
        texture.deallocate()
    glFinish()
    v = fluid._velocity.read_to_numpy()
    assert v is not None
    fluid.deallocate()
    return v[..., :3].astype(np.float64)


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--strength', type=float, default=2.0)
    parser.add_argument('--radius', type=float, default=0.02)
    parser.add_argument('--small', type=float, default=0.006)
    args = parser.parse_args()
    mh.context()
    e, r, small = args.strength, args.radius, args.small

    def config(e1: float, r1: float, e2: float, r2: float, e3: float = 0.0, r3: float = 0.02) -> list[str]:
        return [f'vorticity.strength_1={e1}', f'vorticity.radius_1={r1}',
                f'vorticity.strength_2={e2}', f'vorticity.radius_2={r2}',
                f'vorticity.strength_3={e3}', f'vorticity.radius_3={r3}']

    one = run(config(e, r, 0.0, small), args.seconds)
    three = run(config(0.0, r, 0.0, small, e, r), args.seconds)
    split = run(config(e / 4, r, e / 4, r, e / 2, r), args.seconds)
    sizes = run(config(e, r, e, small), args.seconds)
    still = run(config(0.0, r, 0.0, small), args.seconds)
    scale = float(np.abs(one).max())
    print(f'velocity max {scale:.4f} widths/s')
    print(f'3 alone against 1 alone: max diff {np.abs(three - one).max() / scale:.2e} of the max')
    print(f'split against 1 alone:   max diff {np.abs(split - one).max() / scale:.2e} of the max')
    print(f'1 alone against none:    mean |diff| {np.abs(one - still).mean() / scale:.2e} of the max')
    print(f'two sizes against 1:     mean |diff| {np.abs(sizes - one).mean() / scale:.2e} of the max')


if __name__ == '__main__':
    main()
