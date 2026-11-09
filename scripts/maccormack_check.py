"""MacCormack against semi-Lagrangian on one input: the fluid at deep_flow's preset.

`--test swing` (the default): a disc of velocity (the body) sweeps left and right through the water for `--move`
seconds, then stops; the fluid runs on for the rest of `--seconds`. Each half second it prints, per advection, the
water's top and mean speed (widths/s), in the two cells next to the walls and inside them, and its non-finite count.

`--test swirl`: no body; the water starts as a swirl about the box's z axis (a Rankine vortex, 0.5 widths/s at
its edge) and a cube of fully dyed density beside its center, no fade, no evaporation, no confinement, and runs
`--seconds`. Each second it prints, per advection, the swirl kept (the mean speed over its start), the density's
total and peak (the compression: water squeezed or stretched gathers or thins it), its edge (the 99.5% gradient
over the max, per texel) and the floor: the share of a cell's flux its faces leave unbalanced after the
projection, mean over the cells inside, by the grid's own stencil (the faces' where the fluid has them).

Settings can be changed by path: `--set velocity.vorticity=0.3`.

    python scripts/maccormack_check.py [--test swing|swirl] [--seconds 10] [--move 3] [--set path=value ...]
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_FLOAT, GL_RGBA, GL_TEXTURE_3D, glBindTexture, glFinish, glTexSubImage3D  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.flow import Advection, Fluid3D, Fluid3DSettings  # noqa: E402

PRESET = Path(__file__).resolve().parent.parent / 'apps' / 'deep_flow' / 'data' / 'settings' / 'settings.json'
W, H = 448, 800         # the bridge's output size the body is drawn at
FPS = 60.0


def preset_fluid() -> dict:
    """The fluid's group of deep_flow's preset."""
    def find(node):
        if isinstance(node, dict):
            if 'input' in node and 'velocity' in node and 'density' in node:
                return node
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        return None
    return find(json.loads(PRESET.read_text(encoding='utf-8')))


def body(t: float, moving: bool) -> np.ndarray:
    """The body's velocity (widths/s): a disc swinging left and right, 0 once it stops."""
    values = np.zeros((H, W, 4), np.float32)
    if not moving:
        return values
    speed = 1.5 * np.cos(2.0 * np.pi * t / 1.5)              # widths/s, a full swing every 1.5 s
    cx = W * (0.5 + 0.25 * np.sin(2.0 * np.pi * t / 1.5))
    y, x = np.mgrid[0:H, 0:W]
    disc = (x - cx) ** 2 + (y - H * 0.5) ** 2 < (W * 0.12) ** 2
    values[disc, 0] = speed
    values[disc, 2] = 0.3 * speed                            # a little toward the camera too
    return values


def fluid_settings(advection: Advection, overrides: list[str]) -> Fluid3DSettings:
    """deep_flow's preset fluid, its advection and the settings changed by path."""
    settings = Fluid3DSettings()
    settings.update_from_dict(preset_fluid())
    settings.advection = advection
    for item in overrides:
        path, value = item.split('=', 1)
        target = settings
        *groups, name = path.split('.')
        for group in groups:
            target = getattr(target, group)
        current = getattr(target, name)
        if isinstance(current, bool):
            setattr(target, name, value in ('1', 'true', 'True'))
        elif hasattr(type(current), '__members__'):         # an enum, by its name
            setattr(target, name, type(current)[value])
        else:
            setattr(target, name, type(current)(float(value)))
    return settings


def obstacle_mask() -> np.ndarray:
    """A disc below the body's swing, an obstacle through every layer."""
    values = np.zeros((H, W, 4), np.float32)
    y, x = np.mgrid[0:H, 0:W]
    values[(x - W * 0.5) ** 2 + (y - H * 0.7) ** 2 < (W * 0.1) ** 2] = 1.0
    return values


def run(advection: Advection, seconds: float, move: float, overrides: list[str],
        obstacle: bool = False) -> list[tuple[float, float, float, int, float, float]]:
    settings = fluid_settings(advection, overrides)
    fluid = Fluid3D(settings)
    fluid.allocate(2160, 3840)
    fluid.update(1.0 / FPS)
    solid: np.ndarray | None = None
    if obstacle:
        mask = mh.texture_of(obstacle_mask())
        fluid.set_obstacle(mask)
        fluid.update(1.0 / FPS)
        glFinish()
        cells = fluid._obstacle.read_to_numpy()
        assert cells is not None
        solid = cells.astype(np.float32) > (0.5 * 255 if cells.dtype == np.uint8 else 0.5)
        print(f'{advection.name}: {int(solid.sum())} solid cells', flush=True)
    rows: list[tuple[float, float, float, int, float, float]] = []
    steps = int(seconds * FPS)
    for i in range(steps):
        t = i / FPS
        texture = mh.texture_of(body(t, t < move))
        fluid.drag_velocity(texture)
        fluid.update(1.0 / FPS)
        texture.deallocate()
        if (i + 1) % int(FPS / 2) == 0:
            glFinish()
            v = fluid._velocity.read_to_numpy()
            assert v is not None
            bad = int((~np.isfinite(v[..., :3])).sum())
            speed = np.linalg.norm(np.nan_to_num(v[..., :3]), axis=-1)
            wall = np.zeros(speed.shape, bool)               # the two cells next to any side of the box
            wall[:2], wall[-2:], wall[:, :2], wall[:, -2:], wall[:, :, :2], wall[:, :, -2:] = (True,) * 6
            if solid is not None:                            # with an obstacle the wall column is its cells' speed
                wall = solid
            rows.append((t + 1.0 / FPS, float(speed.max()), float(speed.mean()), bad, float(speed[wall].mean()), float(speed[~wall].mean())))
    fluid.deallocate()
    return rows


def upload(volume, values: np.ndarray) -> None:
    """`values` (depth, height, width, 4) into the RGBA volume's current buffer."""
    d, h, w, _ = values.shape
    glBindTexture(GL_TEXTURE_3D, volume.tex_id)
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, w, h, d, GL_RGBA, GL_FLOAT, np.ascontiguousarray(values, np.float32))
    glBindTexture(GL_TEXTURE_3D, 0)


def swirl_velocity(px: np.ndarray, py: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """The swirl's x and y (widths/s, y down the screen) at `px`, `py` (widths): a Rankine vortex about the box's
    middle, 0.5 widths/s at its edge 0.25 widths out."""
    dx, dy = px - 0.5, py - 0.5 * h / w
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    radius = 0.25
    speed = 0.5 * np.where(r < radius, r / radius, radius / r)
    return -speed * dy / r, speed * dx / r


def swirl_faces(w: int, h: int, d: int) -> np.ndarray:
    """The swirl on the faces (d + 1, h + 1, w + 1, 4): a texel's x the face on its cell's left, y its bottom."""
    z, y, x = np.mgrid[0:d + 1, 0:h + 1, 0:w + 1].astype(np.float32)
    faces = np.zeros((d + 1, h + 1, w + 1, 4), np.float32)
    faces[..., 0] = swirl_velocity(x / w, (y + 0.5) / w, w, h)[0]
    faces[..., 1] = swirl_velocity((x + 0.5) / w, y / w, w, h)[1]
    faces[:, :, 0, 0] = faces[:, :, w, 0] = 0.0                # the closed walls' faces
    faces[:, 0, :, 1] = faces[:, h, :, 1] = 0.0
    return faces


def swirl_start(w: int, h: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """The swirl's velocity at the cells (widths/s, y down the screen) and the density cube, (d, h, w, 4) each."""
    z, y, x = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    px, py = (x + 0.5) / w, (y + 0.5) / w                   # widths
    cx, cy = 0.5, 0.5 * h / w
    velocity = np.zeros((d, h, w, 4), np.float32)
    velocity[..., 0], velocity[..., 1] = swirl_velocity(px, py, w, h)
    density = np.zeros((d, h, w, 4), np.float32)
    cube = (np.abs(px - (cx + 0.15)) < 0.08) & (np.abs(py - cy) < 0.08) & (np.abs(z - d / 2) < d / 4)
    density[cube] = 1.0
    return velocity, density


def floor(v: np.ndarray) -> float:
    """The share of a cell's flux its six faces leave unbalanced, mean over the cells inside, on the centered
    grid: a face's flux the mean of its two cells' velocity, so the imbalance is the central-difference
    divergence the projection makes 0 on its own stencil only approximately."""
    inner = (slice(1, -1),) * 3
    net = np.zeros(v.shape[:3], np.float32)
    total = np.zeros(v.shape[:3], np.float32)
    for axis, channel in ((2, 0), (1, 1), (0, 2)):          # x along the width, y the height, z the depth
        component = v[..., channel]
        after = 0.5 * (component + np.roll(component, -1, axis))
        before = np.roll(after, 1, axis)
        net += after - before
        total += np.abs(after) + np.abs(before)
    net, total = np.abs(net[inner]), total[inner]
    moving = total > 1e-3 * float(total.max())
    return float(net[moving].sum() / max(float(total[moving].sum()), 1e-12))


def face_floor(faces: np.ndarray) -> float:
    """The floor on the staggered grid: `faces` (depth + 1, height + 1, width + 1, 4), a texel's x, y, z the faces
    on its cell's left, bottom and front; per cell its six faces' imbalance over their summed size."""
    f = faces.astype(np.float32)
    x0, x1 = f[:-1, :-1, :-1, 0], f[:-1, :-1, 1:, 0]
    y0, y1 = f[:-1, :-1, :-1, 1], f[:-1, 1:, :-1, 1]
    z0, z1 = f[:-1, :-1, :-1, 2], f[1:, :-1, :-1, 2]
    net = np.abs(x1 - x0 + y1 - y0 + z1 - z0)
    total = np.abs(x0) + np.abs(x1) + np.abs(y0) + np.abs(y1) + np.abs(z0) + np.abs(z1)
    inner = (slice(1, -1),) * 3
    net, total = net[inner], total[inner]
    moving = total > 1e-3 * float(total.max())
    return float(net[moving].sum() / max(float(total[moving].sum()), 1e-12))


def edge(alpha: np.ndarray) -> float:
    """The density's edge: the 90% of its gradient's size per texel over its max, among the texels it dyes
    (above 1% of its max), 0.5 for a cube's sharp step."""
    gz, gy, gx = np.gradient(alpha)
    size = np.sqrt(gx * gx + gy * gy + gz * gz)
    dyed = alpha > 0.01 * float(alpha.max())
    return float(np.percentile(size[dyed], 90.0) / max(float(alpha.max()), 1e-6)) if dyed.any() else 0.0


def run_swirl(advection: Advection, seconds: float, overrides: list[str]) -> list[tuple[float, ...]]:
    calm = ['velocity.vorticity=0', 'density.fade_time=0', 'density.evaporation_time=0',
            'velocity.fade_time=0', 'velocity.evaporation_time=0']
    fluid = Fluid3D(fluid_settings(advection, calm + overrides))
    fluid.allocate(2160, 3840)
    fluid.update(1.0 / FPS)
    velocity_volume = fluid._velocity
    density_volume = fluid._simulation_density._density
    velocity, density = swirl_start(velocity_volume.width, velocity_volume.height, velocity_volume.depth)
    upload(velocity_volume, velocity)
    if hasattr(fluid, '_faces'):
        upload(fluid._faces.volume, swirl_faces(velocity_volume.width, velocity_volume.height, velocity_volume.depth))
    upload(density_volume, density)
    start_speed = float(np.linalg.norm(velocity[..., :3], axis=-1).mean())
    rows: list[tuple[float, ...]] = []
    for i in range(int(seconds * FPS) + 1):
        if i % int(FPS) == 0:
            glFinish()
            v = velocity_volume.read_to_numpy()
            a = density_volume.read_to_numpy()
            assert v is not None and a is not None
            v = v.astype(np.float32)
            alpha = a[..., 3].astype(np.float32)
            speed = float(np.linalg.norm(v[..., :3], axis=-1).mean())
            faces = fluid._faces.volume.read_to_numpy() if hasattr(fluid, '_faces') else None
            unbalanced = (face_floor(faces) if faces is not None else floor(v)) if i else float('nan')
            rows.append((i / FPS, speed / start_speed, float(alpha.sum()), float(alpha.max()), edge(alpha), unbalanced))
        fluid.update(1.0 / FPS)
    fluid.deallocate()
    return rows


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--test', choices=('swing', 'swirl'), default='swing')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--move', type=float, default=3.0, help='seconds the body moves')
    parser.add_argument('--set', action='append', default=[], help='a setting by path, e.g. velocity.vorticity=0.3')
    parser.add_argument('--obstacle', action='store_true', help='swing: a disc obstacle below the body; the wall column is its cells')
    args = parser.parse_args()
    mh.context()
    if args.test == 'swirl':
        swirl = {advection: run_swirl(advection, args.seconds, args.set) for advection in (Advection.SEMI_LAGRANGIAN, Advection.MACCORMACK)}
        print(f"{'t s':>5}  {'':>2} {'swirl':>6} {'total':>10} {'peak':>6} {'edge':>6} {'floor':>8}")
        for advection, label in ((Advection.SEMI_LAGRANGIAN, 'SL'), (Advection.MACCORMACK, 'MC')):
            for t, kept, total, peak, sharp, unbalanced in swirl[advection]:
                print(f'{t:5.1f}  {label:>2} {kept:6.3f} {total:10.1f} {peak:6.3f} {sharp:6.3f} {unbalanced:8.5f}')
        return
    results = {advection: run(advection, args.seconds, args.move, args.set, args.obstacle) for advection in (Advection.SEMI_LAGRANGIAN, Advection.MACCORMACK)}
    print(f"{'t s':>5}  {'SL top':>8} {'SL mean':>8} {'SL wall':>8} {'SL inner':>8}  {'MC top':>8} {'MC mean':>8} {'MC wall':>8} {'MC inner':>8} {'MC bad':>7}")
    for sl, mc in zip(results[Advection.SEMI_LAGRANGIAN], results[Advection.MACCORMACK]):
        print(f'{sl[0]:5.1f}  {sl[1]:8.3f} {sl[2]:8.4f} {sl[4]:8.4f} {sl[5]:8.4f}  {mc[1]:8.3f} {mc[2]:8.4f} {mc[4]:8.4f} {mc[5]:8.4f} {mc[3]:7d}')


if __name__ == '__main__':
    main()
