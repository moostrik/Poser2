"""Play saved frames through the Lucas–Kanade flow and the velocity trail bridge, A against B, and measure both.

    python scripts/velocity_trail_compare.py FRAMES [--a NAME=VALUE ...] [--b NAME=VALUE ...]

FRAMES is what depth_track_capture saved from one loop of a recording (temp/depth_track/RECORDING). No app
and no camera: a hidden window gives the GL context, one `LucasKanadeFlow` measures the saved frames with the
settings they were recorded with, and two `VelocityTrailBridge`, A and B, take its velocity; `--a` and
`--b` set any of the trail's settings (pivot, curve, limit, z_limit, attack, smooth, blur_radius), the rest
as in deep_flow's preset. One loop, as fast as it goes; the trail steps once per flow frame, by its interval.

Out comes one table per side, in widths per second, on the body (the mask, 12 px and more inside):

    Noise         the still part's p99, at the edge (0–6 px inside) and inside
    Slow, Fast    per move half, the mean over the pixels that move in the loop
    Fast ÷ slow   per move; for the push also per distance: the push over the flow's own speed
    Fast ÷ noise  the fast half's mean over the still part's p99 inside
    Through       per move half, the trail's peak over the flow's peak (what the Ema lets through)
    GPU           the trail's time per frame
"""

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import glFinish

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.gl import Texture  # noqa: E402
from modules.flow import LucasKanadeFlow, LucasKanadeFlowSettings, VelocityTrailBridge, VelocityTrailBridgeSettings  # noqa: E402
from depth_track_compare import Frames, context, read_back, recorded_settings, upload  # noqa: E402

SETTLE: float = 1.5      # s of the still part left out of its noise: the trail's memory of the flow's first frames
PRESET: Path =Path(__file__).resolve().parent.parent / 'apps' / 'deep_flow' / 'data' / 'settings' / 'settings.json'


def trail_settings(items: list[str]) -> tuple[VelocityTrailBridgeSettings, dict[str, float]]:
    """The preset's trail settings with `items` (NAME=VALUE) set on them."""
    settings = VelocityTrailBridgeSettings()
    settings.update_from_dict(json.loads(PRESET.read_text())['render']['flow']['flow_bridge']['velocity_trail'])
    changed: dict[str, float] = {}
    for item in items:
        name, value = item.split('=', 1)
        if name not in settings.fields:
            raise SystemExit(f'unknown setting {name}: one of {", ".join(settings.fields)}')
        changed[name] = float(value)
        setattr(settings, name, changed[name])
    return settings, changed


class Measures:
    """One side's measures over the loop."""

    def __init__(self) -> None:
        self.noise: dict[str, list[np.ndarray]] = {'xy edge': [], 'xy inside': [], 'z edge': [], 'z inside': [], 'flow xy': [], 'flow z': []}
        self.segments: list[str] = []
        self.xy: list[float] = []           # the trail's |x/y| on the moving body pixels
        self.z: list[float] = []            # the trail's |z| there
        self.flow_xy: list[float] = []      # the flow's |x/y| there
        self.flow_z: list[float] = []
        self.milliseconds: list[float] = []

    def add(self, segment: str, trail: np.ndarray, flow: np.ndarray, core: np.ndarray, edge: np.ndarray, moving: np.ndarray, milliseconds: float) -> None:
        xy, z = np.hypot(trail[..., 0], trail[..., 1]), np.abs(trail[..., 2])
        self.segments.append(segment)
        self.milliseconds.append(milliseconds)
        self.xy.append(float(np.mean(xy[moving])) if moving.any() else 0.0)
        self.z.append(float(np.mean(z[moving])) if moving.any() else 0.0)
        self.flow_xy.append(float(np.mean(np.hypot(flow[..., 0], flow[..., 1])[moving])) if moving.any() else 0.0)
        self.flow_z.append(float(np.mean(np.abs(flow[..., 2])[moving])) if moving.any() else 0.0)
        if segment == 'still':
            self.noise['xy edge'].append(xy[edge]); self.noise['xy inside'].append(xy[core])
            self.noise['z edge'].append(z[edge]); self.noise['z inside'].append(z[core])
            self.noise['flow xy'].append(np.hypot(flow[..., 0], flow[..., 1])[core]); self.noise['flow z'].append(np.abs(flow[..., 2])[core])

    def report(self, moves: int) -> str:
        segments = np.array(self.segments)
        p99 = {name: float(np.percentile(np.concatenate(parts), 99)) if parts else 0.0 for name, parts in self.noise.items()}
        lines = [f'{"":10s} {"Noise edge":>10s} {"Noise in":>9s} {"Slow":>7s} {"Fast":>7s} {"F÷S":>6s} {"F÷S/dist":>8s} {"F÷noise":>8s} '
                 f'{"Thru slow":>9s} {"Thru fast":>9s}']
        for n in range(1, moves + 1):
            slow, fast = segments == f'move {n} slow', segments == f'move {n} fast'
            if slow.sum() < 3 or fast.sum() < 3:
                continue
            for axis, trail, flow in (('xy', self.xy, self.flow_xy), ('z', self.z, self.flow_z)):
                t, f = np.array(trail), np.array(flow)
                s_mean, f_mean = float(np.mean(t[slow])), float(np.mean(t[fast]))
                per_distance = (f_mean / max(float(np.mean(f[fast])), 1e-9)) / max(s_mean / max(float(np.mean(f[slow])), 1e-9), 1e-9)
                through_slow = float(np.max(t[slow])) / max(float(np.max(f[slow])), 1e-9)
                through_fast = float(np.max(t[fast])) / max(float(np.max(f[fast])), 1e-9)
                lines.append(f'move {n} {axis:3s} {p99[axis + " edge"]:10.3f} {p99[axis + " inside"]:9.3f} {s_mean:7.3f} {f_mean:7.3f} '
                             f'{f_mean / max(s_mean, 1e-9):6.2f} {per_distance:8.2f} {f_mean / max(p99[axis + " inside"], 1e-9):8.1f} '
                             f'{through_slow:9.2f} {through_fast:9.2f}')
        still = [np.concatenate(self.noise[name]) for name in ('flow xy', 'flow z')]
        lines.append('flow still inside: ' + ', '.join(f'{axis} >0 {np.mean(v > 0):.4f} p99 {np.percentile(v, 99):.3f} p99.9 {np.percentile(v, 99.9):.3f} max {np.max(v):.3f}'
                                                        for axis, v in zip(('xy', 'z'), still)))
        lines.append(f'GPU {np.median(self.milliseconds):.3f} ms per frame')
        return '\n    '.join(lines)


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path)
    parser.add_argument('--a', action='append', default=[], metavar='NAME=VALUE', help="set one of A's trail settings")
    parser.add_argument('--b', action='append', default=[], metavar='NAME=VALUE', help="set one of B's trail settings")
    args = parser.parse_args()
    frames = Frames(args.frames)
    w, h = frames.width, frames.height
    context(False, 64, 64)

    flow_settings = LucasKanadeFlowSettings()
    flow_settings.update_from_dict(recorded_settings(frames.calls[0]['settings']))
    flow = LucasKanadeFlow(flow_settings)
    flow.allocate(w, h)
    sides: list[tuple[str, VelocityTrailBridge, dict[str, float], Measures]] = []
    for name, items in (('A', args.a), ('B', args.b)):
        settings, changed = trail_settings(items)
        trail = VelocityTrailBridge(settings)
        trail.allocate(w, h)
        sides.append((name, trail, changed, Measures()))
    print(frames.describe())
    for name, _, changed, _ in sides:
        print(f'{name}: {changed or "as the preset"}')

    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    for k in range(len(frames)):
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask.astype(np.float32))
        c = frames.calls[k]
        flow.set_input(luminance)
        flow.set_mask(mask_texture)
        flow.update(depth_texture, c['delta_time'], c['view_span'], tuple(c['depth_step']))
        velocity = read_back(flow.velocity)[..., :3]
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        core, edge = mask & (depth > 0) & (inside > 12), mask & (depth > 0) & (inside <= 6)
        moving = core & frames.moved
        segment = frames.segment(k)
        if segment == 'still' and frames.t[k] < frames.still[0] + SETTLE:
            segment = ''                          # the trails still let go of the flow's start
        for _, trail, _, measures in sides:
            glFinish()
            began = time.perf_counter()
            trail.set_velocity(flow.velocity)
            trail.update(c['delta_time'])
            glFinish()
            milliseconds = (time.perf_counter() - began) * 1000.0
            measures.add(segment, read_back(trail.smooth_velocity)[..., :3], velocity, core, edge, moving, milliseconds)

    for name, _, _, measures in sides:
        print(f'{name}: {measures.report(len(frames.moves))}', flush=True)
    glfw.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
