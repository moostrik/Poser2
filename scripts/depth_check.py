"""How the 3D fluid moves in depth, played on saved frames: the water's z speed against its xy speed, the
body's own z, and what the projection leaves of the divergence.

    python scripts/depth_check.py FRAMES [--side NAME[:PATH=VALUE,...]] ... [--every N] [--output W H]

FRAMES and the sides as `fluid_compare.py` takes them (deep_flow's preset, a side's PATH=VALUE under `bridge.`
or `fluid.`). Every `--every` steps, over the loop's fast half, the whole velocity volume is read back:

    xy        the water's |(u, v)| mean over the cells that move (|velocity| above 0.01 widths/s), widths/s
    z         the water's |w| mean over the same cells, widths/s
    z / xy    how much of the water's motion is in depth
    body z    the body's |z| mean where it moves (the boundary velocity's z), widths/s, as it goes in
    body z/xy the body's own share in depth, to set z / xy against
    residual  the divergence after the projection, its mean |·| over the cells, per second (widths/s per width)

Then the step's GPU time, ms, mean and p95.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import GL_RGBA, GL_RED, GL_FLOAT, GL_TEXTURE_3D, glBindTexture, glFinish, glGetTexImage, glMemoryBarrier, GL_ALL_BARRIER_BITS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.gl import Texture, Texture3D  # noqa: E402
from modules.flow import LucasKanadeFlow3D, LucasKanadeFlow3DSettings  # noqa: E402
from fluid_compare import Side, windows  # noqa: E402
from depth_velocity_compare import Frames, context, read_back, recorded_settings, upload  # noqa: E402

MOVING: float = 0.01        # widths/s: a cell's speed above which it counts as moving


def volume(texture: Texture3D, channels: int) -> np.ndarray:
    """A 3D texture as float32 (D, H, W, channels)."""
    glMemoryBarrier(GL_ALL_BARRIER_BITS)
    glBindTexture(GL_TEXTURE_3D, texture.tex_id)
    data = glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA if channels == 4 else GL_RED, GL_FLOAT)
    glBindTexture(GL_TEXTURE_3D, 0)
    return np.frombuffer(data, np.float32).reshape(texture.depth, texture.height, texture.width, channels)


def residual(fluid) -> float:
    """The divergence of the projected velocity, its mean |·|: the divergence pass run again after the step."""
    inner = fluid._fluid
    inner._divergence_shader.use(inner._faces.volume, inner._obstacle, inner._divergence_vol, inner._cell,
                                 has_obstacles=inner._has_obstacles, closed=inner._closed)
    glFinish()
    return float(np.mean(np.abs(volume(inner._divergence_vol, 1))))


def main() -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path)
    parser.add_argument('--side', action='append', default=[], metavar='NAME[:PATH=VALUE,...]')
    parser.add_argument('--every', type=int, default=8, help='read the volume back every N steps of the fast half')
    parser.add_argument('--output', type=int, nargs=2, default=(1080, 1920), metavar=('W', 'H'))
    args = parser.parse_args()
    frames = Frames(args.frames)
    context(False, 64, 64)
    flow_settings = LucasKanadeFlow3DSettings()
    flow_settings.update_from_dict(recorded_settings(frames.calls[0]['settings']))
    flow = LucasKanadeFlow3D(flow_settings)
    flow.allocate(frames.width, frames.height)
    sides = [Side(spec, tuple(args.output)) for spec in (args.side or ['preset'])]
    fast = windows(frames)['fast']
    rows: dict[str, dict[str, list[float]]] = {s.name: {k: [] for k in ('xy', 'z', 'body_xy', 'body_z', 'residual')} for s in sides}
    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    for k in range(len(frames)):
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask.astype(np.float32))
        c = frames.calls[k]
        flow.set_input(luminance)
        flow.set_mask(mask_texture)
        flow.update(depth_texture, c['delta_time'], c['view_span'], tuple(c['depth_step']))
        for side in sides:
            side.step(flow.velocity, mask_texture, c['delta_time'])
            if not fast[k] or k % args.every:
                continue
            row = rows[side.name]
            velocity = volume(side.fluid.ports['velocity'].texture, 4)
            speed = np.linalg.norm(velocity[..., :3], axis=-1)
            moving = speed > MOVING
            if moving.any():
                row['xy'].append(float(np.mean(np.hypot(velocity[..., 0], velocity[..., 1])[moving])))
                row['z'].append(float(np.mean(np.abs(velocity[..., 2])[moving])))
            body = read_back(side.bridge.output.boundary_velocity)
            body_speed = np.linalg.norm(body[..., :3], axis=-1)
            there = body_speed > MOVING
            if there.any():
                row['body_xy'].append(float(np.mean(np.hypot(body[..., 0], body[..., 1])[there])))
                row['body_z'].append(float(np.mean(np.abs(body[..., 2])[there])))
            row['residual'].append(residual(side.fluid))
    print('| Side | xy | z | z / xy | body z | body z / xy | residual | GPU ms mean / p95 |')
    print('|---|---|---|---|---|---|---|---|')
    for side in sides:
        r = {key: float(np.mean(v)) if v else 0.0 for key, v in rows[side.name].items()}
        ms = np.array(side.milliseconds[10:])
        print(f"| {side.name} | {r['xy']:.4f} | {r['z']:.4f} | {r['z'] / max(r['xy'], 1e-9):.2f} | {r['body_z']:.4f} | "
              f"{r['body_z'] / max(r['body_xy'], 1e-9):.2f} | {r['residual']:.4f} | {ms.mean():.2f} / {np.percentile(ms, 95):.2f} |", flush=True)
    glfw.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
