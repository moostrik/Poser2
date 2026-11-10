"""The fluid's isotropy over stretched cells: a swirl in the image plane (about z) and one in depth (about y),
the same size in widths, left alone; whether the one in depth keeps its round shape and its speed as the one
across does, and what the projection leaves of the divergence.

    python scripts/isotropy_check.py [--seconds 2] [--set path=value ...]

deep_flow's preset (`maccormack_check.fluid_settings`), still water, no vorticity, no fade: each swirl a Rankine
vortex 0.1 widths in radius, 0.3 widths/s at its edge, about the box's middle. Each half second, per swirl:

    kept      its mean speed over the start's
    aspect    the spread of its speed along its two axes, in widths, the first over the second: 1 round
    residual  the divergence's mean |·| after the step's projection, per second
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import OpenGL
OpenGL.ERROR_CHECKING = False
import numpy as np
from OpenGL.GL import GL_RED, GL_RGBA, GL_FLOAT, GL_TEXTURE_3D, glBindTexture, glFinish, glGetTexImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitor_harness as mh  # noqa: E402
import maccormack_check as mc  # noqa: E402
from modules.flow import Advection, Fluid3D  # noqa: E402

FPS: float = 60.0
RADIUS: float = 0.1         # widths
EDGE_SPEED: float = 0.3     # widths/s


def read(texture, channels: int) -> np.ndarray:
    glBindTexture(GL_TEXTURE_3D, texture.tex_id)
    data = glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA if channels == 4 else GL_RED, GL_FLOAT)
    glBindTexture(GL_TEXTURE_3D, 0)
    return np.frombuffer(data, np.float32).reshape(texture.depth, texture.height, texture.width, channels).copy()


def rankine(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The swirl's two components at offsets `a`, `b` (widths) from its axis."""
    r = np.sqrt(a * a + b * b) + 1e-9
    speed = EDGE_SPEED * np.where(r < RADIUS, r / RADIUS, RADIUS / r)
    return -speed * b / r, speed * a / r


def swirl_faces(cells: tuple[int, int, int], cell: tuple[float, float, float], plane: str) -> np.ndarray:
    """The swirl on the faces (d + 1, h + 1, w + 1, 4), positions in widths: a texel's x, y, z the faces on its
    cell's left, bottom and front. `plane` 'xy' about z, 'xz' about y."""
    w, h, d = cells
    z, y, x = np.mgrid[0:d + 1, 0:h + 1, 0:w + 1].astype(np.float64)
    center = (0.5 * w * cell[0], 0.5 * h * cell[1], 0.5 * d * cell[2])
    faces = np.zeros((d + 1, h + 1, w + 1, 4), np.float32)
    def at(fx, fy, fz):       # a face's place in widths, from its index's offsets
        return fx * cell[0] - center[0], fy * cell[1] - center[1], fz * cell[2] - center[2]
    if plane == 'xy':
        px, py, _ = at(x, y + 0.5, z + 0.5)
        faces[..., 0] = rankine(px, py)[0]
        px, py, _ = at(x + 0.5, y, z + 0.5)
        faces[..., 1] = rankine(px, py)[1]
    else:
        px, _, pz = at(x, y + 0.5, z + 0.5)
        faces[..., 0] = rankine(px, pz)[0]
        px, _, pz = at(x + 0.5, y + 0.5, z)
        faces[..., 2] = rankine(px, pz)[1]
    faces[:, :, 0, 0] = faces[:, :, w, 0] = 0.0
    faces[:, 0, :, 1] = faces[:, h, :, 1] = 0.0
    faces[0, :, :, 2] = faces[d, :, :, 2] = 0.0
    return faces


def measure(fluid: Fluid3D, plane: str) -> tuple[float, float]:
    """The swirl's mean speed and its aspect: its speed's spread along its two axes, in widths."""
    v = read(fluid._velocity, 4)
    d, h, w, _ = v.shape
    cell = fluid._cell
    z, y, x = np.mgrid[0:d, 0:h, 0:w].astype(np.float64)
    px, py, pz = (x + 0.5 - 0.5 * w) * cell[0], (y + 0.5 - 0.5 * h) * cell[1], (z + 0.5 - 0.5 * d) * cell[2]
    speed = np.linalg.norm(v[..., :3], axis=-1).astype(np.float64)
    mass = max(float(speed.sum()), 1e-12)
    first, second = (px, py) if plane == 'xy' else (px, pz)
    spread = [float(np.sqrt((speed * a * a).sum() / mass)) for a in (first, second)]
    return float(speed.mean()), spread[0] / max(spread[1], 1e-12)


def residual(fluid: Fluid3D) -> float:
    fluid._divergence_shader.use(fluid._faces.volume, fluid._obstacle, fluid._divergence_vol, fluid._cell,
                                 has_obstacles=fluid._has_obstacles, closed=fluid._closed)
    glFinish()
    return float(np.mean(np.abs(read(fluid._divergence_vol, 1))))


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    mh.context()
    calm = ['velocity.vorticity=0', 'velocity.fade_time=0', 'velocity.evaporation_time=0', 'velocity.viscosity=0']
    print(f"{'plane':>5} {'t s':>5} {'kept':>6} {'aspect':>7} {'residual':>9}")
    for plane in ('xy', 'xz'):
        fluid = Fluid3D(mc.fluid_settings(Advection.SEMI_LAGRANGIAN, calm + args.set))
        fluid.allocate(1080, 1920)
        fluid.update(1.0 / FPS)
        faces = swirl_faces(fluid._cells, fluid._cell, plane)
        mc.upload(fluid._faces.volume, faces)
        fluid._faces_to_centers()
        glFinish()
        start, _ = measure(fluid, plane)
        for i in range(int(args.seconds * FPS) + 1):
            if i % int(FPS / 2) == 0:
                speed, aspect = measure(fluid, plane)
                print(f'{plane:>5} {i / FPS:5.1f} {speed / max(start, 1e-12):6.3f} {aspect:7.3f} {residual(fluid) if i else float("nan"):9.4f}', flush=True)
            fluid.update(1.0 / FPS)
        print(f'{plane}: {fluid._cells} cells, a cell {tuple(round(c, 5) for c in fluid._cell)} widths')
        fluid.deallocate()


if __name__ == '__main__':
    main()
