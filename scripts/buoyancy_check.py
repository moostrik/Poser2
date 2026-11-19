"""The buoyancy on a still blob in a closed box: its force, and the water's first step after the projection.

The fluid at deep_flow's preset, every wall closed, the water at rest. A sphere `--radius` widths across at the
box's middle holds a uniform warmth (`--warmth`, `temperature.buoyancy` `--buoyancy`), or, with the warmth 0, a
uniform density of amount `--amount` (`density.sink` `--sink`, the temperature off), its color in RGB or in Oklab (a and b
negative). One buoyancy pass, then one projection. It prints:

- the force at the cells against the textbook's, gravity × (sink × amount − buoyancy × warmth): its largest
  error over every cell for the warmth, which is on the force's grid; for the density, on its own grid and read
  by uv, over the cells two density texels or more inside or outside the sphere's surface;
- the faces the one buoyancy pass makes against those the force at the cells makes, added in a second pass (the
  force held at half precision there): their largest difference;
- the water's velocity at the sphere's middle (its cells within half the radius) along gravity, after the
  projection, over the force × dt it was given: a sphere in still water takes 2/3 of it, the rest pushing the
  water around it (its added mass, half its own) **(deduction)**; once at the preset's `pressure.cycles` and once
  at `--cycles`.

It runs on the GPU like the apps: run it only while no app is running.

    python scripts/buoyancy_check.py [--radius 0.15] [--warmth 0.75] [--buoyancy 2] [--amount 0.5] [--sink 3]
                                     [--cycles 8] [--set path=value ...]
"""

import OpenGL
OpenGL.ERROR_CHECKING = False  # as launcher.py runs the app; a glGetError per call otherwise makes the CPU the bottleneck
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_FLOAT, GL_RED, GL_RGBA, GL_TEXTURE_3D, glBindTexture, glFinish, glTexSubImage3D  # noqa: E402

import monitor_harness as mh  # noqa: E402
import maccormack_check as mc  # noqa: E402
from modules.flow import Advection, Fluid3D  # noqa: E402

OUTPUT: tuple[int, int] = (448, 800)     # the composite's size; nothing here reads it
DENSITY_EDGE: float = 2.0                 # density texels from the surface left out of the density's force check


def upload(volume, values: np.ndarray) -> None:
    """`values`, (depth, height, width) for one channel or (depth, height, width, 4), into the volume's current buffer."""
    d, h, w = values.shape[:3]
    glBindTexture(GL_TEXTURE_3D, volume.tex_id)
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, w, h, d, GL_RED if values.ndim == 3 else GL_RGBA, GL_FLOAT,
                    np.ascontiguousarray(values, np.float32))
    glBindTexture(GL_TEXTURE_3D, 0)


def read(volume) -> np.ndarray:
    glFinish()
    values = volume.read_to_numpy()
    assert values is not None
    return values.astype(np.float64)


def distance(size: tuple[int, int, int], box: tuple[float, float, float]) -> np.ndarray:
    """Each cell center's distance from the box's middle, widths, (depth, height, width), on a grid of `size`
    (width, height, depth) over a box `box` (x, y, z) widths."""
    w, h, d = size
    z, y, x = np.meshgrid((np.arange(d) + 0.5) / d * box[2], (np.arange(h) + 0.5) / h * box[1],
                          (np.arange(w) + 0.5) / w * box[0], indexing='ij')
    return np.sqrt((x - box[0] / 2) ** 2 + (y - box[1] / 2) ** 2 + (z - box[2] / 2) ** 2)


def fluid_at(overrides: list[str]) -> Fluid3D:
    fluid = Fluid3D(mc.fluid_settings(Advection.SEMI_LAGRANGIAN, ['wall=CLOSED', 'temperature.enabled=1', 'fine_density.enabled=0',
                                                                    'particles.enabled=0', 'composite.light.enabled=0',
                                                                    *overrides]))
    fluid.allocate(*OUTPUT)
    fluid.update(1.0 / mc.FPS)      # the deferred walls applied, dt set
    fluid.reset()
    glFinish()
    return fluid


def box_of(fluid: Fluid3D) -> tuple[float, float, float]:
    return (1.0, 1.0 / fluid._aspect, fluid._z_extent)


def one_pass(fluid: Fluid3D) -> tuple[np.ndarray, float]:
    """The force at the cells (xyz), its monitor's, and the largest difference of the faces the one buoyancy pass
    makes from the faces the force at the cells makes, added in a second pass as `add_force` does, from rest."""
    fluid._faces.clear()
    shown = fluid.ports['buoyancy'].prepare
    assert shown is not None
    shown()                                 # as its monitor: the force at the cells made too
    fluid._apply_buoyancy()
    faces = read(fluid._faces.volume)
    fluid._faces.clear()
    fluid._add_to_faces_shader.add_force(fluid._faces.volume, fluid._buoyancy_force_vol, fluid._dt, fluid._obstacle,
                                         fluid._closed, fluid._has_obstacles)
    two_pass = read(fluid._faces.volume)
    return read(fluid._buoyancy_force_vol)[..., :3], float(np.abs(faces - two_pass).max())


def first_step(fluid: Fluid3D, inside: np.ndarray, expected: float) -> float:
    """The projection's share of the force × dt the sphere's middle keeps along gravity, the faces at rest
    before the buoyancy."""
    fluid._faces.clear()
    fluid._pressure.clear_all()
    fluid._apply_buoyancy()
    fluid._enforce_incompressibility()
    fluid._faces_to_centers()
    velocity = read(fluid._velocity)
    t = fluid.settings.gravity
    gravity = (t.x, t.y, t.z)
    along = sum(velocity[..., i] * g for i, g in enumerate(gravity)) / max(sum(g * g for g in gravity), 1e-12)
    return float(along[inside].mean()) / (expected * fluid._dt)


def warm(args, overrides: list[str]) -> None:
    fluid = fluid_at(['temperature.enabled=1', f'temperature.buoyancy={args.buoyancy}', 'density.sink=0', *overrides])
    try:
        cells = fluid._cells
        r = distance(cells, box_of(fluid))
        warmth = np.where(r < args.radius, args.warmth, 0.0)
        upload(fluid._temperature, warmth)
        force, faces_error = one_pass(fluid)
        t = fluid.settings.gravity
        gravity = np.array((t.x, t.y, t.z))
        textbook = -args.buoyancy * warmth[..., None] * gravity
        error = float(np.abs(force - textbook).max())
        expected = -args.buoyancy * args.warmth
        print(f'warm    force  textbook {expected:+.4f} widths/s² along gravity  largest error {error:.2e} '
              f'over {force.shape[0] * force.shape[1] * force.shape[2]} cells', flush=True)
        print(f'warm    faces  one pass against two  largest difference {faces_error:.2e} widths/s', flush=True)
        middle = r < 0.5 * args.radius
        cycles = fluid.settings.pressure.cycles
        for c in (cycles, args.cycles):
            fluid.settings.pressure.cycles = c
            upload(fluid._temperature, warmth)
            print(f'warm    first step, cycles {c}: velocity / (force × dt) {first_step(fluid, middle, expected):.4f}', flush=True)
    finally:
        fluid.deallocate()


def dense(args, color_space: str, overrides: list[str]) -> None:
    fluid = fluid_at(['temperature.enabled=0', f'density.sink={args.sink}', f'density.color_space={color_space}',
                      *overrides])
    try:
        density = fluid._density._density
        density_size = (density.width, density.height, density.depth)
        box = box_of(fluid)
        r_density = distance(density_size, box)
        amount = np.where(r_density < args.radius, args.amount, 0.0)
        color = (0.6, -0.1, -0.2) if color_space == 'OKLAB' else (0.2, 0.5, 0.9)      # Oklab's a and b below 0
        values = np.zeros((*amount.shape, 4))
        for i, c in enumerate(color):
            values[..., i] = c * amount
        values[..., 3] = amount
        upload(density, values)
        force, faces_error = one_pass(fluid)
        t = fluid.settings.gravity
        gravity = np.array((t.x, t.y, t.z))
        r = distance(fluid._cells, box)
        texel = max(b / n for b, n in zip(box, density_size))
        inside, outside = r < args.radius - DENSITY_EDGE * texel, r > args.radius + DENSITY_EDGE * texel
        textbook = args.sink * np.where(inside, args.amount, 0.0)[..., None] * gravity
        kept = inside | outside
        error = float(np.abs(force - textbook)[kept].max())
        expected = args.sink * args.amount
        print(f'dense   {color_space:<5}  force  textbook {expected:+.4f} widths/s² along gravity  largest error '
              f'{error:.2e} over {int(kept.sum())} cells', flush=True)
        print(f'dense   {color_space:<5}  faces  one pass against two  largest difference {faces_error:.2e} widths/s',
              flush=True)
        middle = r < 0.5 * args.radius
        cycles = fluid.settings.pressure.cycles
        for c in (cycles, args.cycles):
            fluid.settings.pressure.cycles = c
            print(f'dense   {color_space:<5}  first step, cycles {c}: velocity / (force × dt) '
                  f'{first_step(fluid, middle, expected):.4f}', flush=True)
    finally:
        fluid.deallocate()


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--radius', type=float, default=0.15)
    parser.add_argument('--warmth', type=float, default=0.75)
    parser.add_argument('--buoyancy', type=float, default=2.0)
    parser.add_argument('--amount', type=float, default=0.5)
    parser.add_argument('--sink', type=float, default=3.0)
    parser.add_argument('--cycles', type=int, default=8)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    mh.context()
    warm(args, args.set)
    for color_space in ('RGB', 'OKLAB'):
        dense(args, color_space, args.set)


if __name__ == '__main__':
    main()
