"""MacCormack against semi-Lagrangian on one input: the fluid at deep_flow's preset, a body moving through it.

A disc of velocity (the body) sweeps left and right through the water for `--move` seconds, then stops; the fluid
runs on for the rest of `--seconds`. Each half second it prints, per advection, the water's top and mean speed
(widths/s) and its non-finite count, so a growth MacCormack has and the semi-Lagrangian step has not shows as a
column that keeps rising after the body stops. Settings can be changed by path: `--set velocity.vorticity=0.3`.

    python scripts/maccormack_check.py [--seconds 10] [--move 3] [--set path=value ...]
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import glFinish  # noqa: E402

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


def run(advection: Advection, seconds: float, move: float, overrides: list[str]) -> list[tuple[float, float, float, int, float, float]]:
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
    fluid = Fluid3D(settings)
    fluid.allocate(2160, 3840)
    fluid.update(1.0 / FPS)
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
            rows.append((t + 1.0 / FPS, float(speed.max()), float(speed.mean()), bad, float(speed[wall].mean()), float(speed[~wall].mean())))
    fluid.deallocate()
    return rows


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--move', type=float, default=3.0, help='seconds the body moves')
    parser.add_argument('--set', action='append', default=[], help='a setting by path, e.g. velocity.vorticity=0.3')
    args = parser.parse_args()
    mh.context()
    results = {advection: run(advection, args.seconds, args.move, args.set) for advection in (Advection.SEMI_LAGRANGIAN, Advection.MACCORMACK)}
    print(f"{'t s':>5}  {'SL top':>8} {'SL mean':>8} {'SL wall':>8} {'SL inner':>8}  {'MC top':>8} {'MC mean':>8} {'MC wall':>8} {'MC inner':>8} {'MC bad':>7}")
    for sl, mc in zip(results[Advection.SEMI_LAGRANGIAN], results[Advection.MACCORMACK]):
        print(f'{sl[0]:5.1f}  {sl[1]:8.3f} {sl[2]:8.4f} {sl[4]:8.4f} {sl[5]:8.4f}  {mc[1]:8.3f} {mc[2]:8.4f} {mc[4]:8.4f} {mc[5]:8.4f} {mc[3]:7d}')


if __name__ == '__main__':
    main()
