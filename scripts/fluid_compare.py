"""Play saved frames through the Lucas–Kanade flow, the 3D bridge and the 3D fluid, side by side, and measure the fluid.

    python scripts/fluid_compare.py FRAMES [--side NAME[:PATH=VALUE,...]] ... [--every N] [--output W H] [--save DIR]

FRAMES is what depth_velocity_capture saved from one loop of a recording. No app and no camera: a hidden window
gives the GL context, one `LucasKanadeFlow3D` measures the saved frames with the settings they were saved with,
and every side is a `Bridge3DLayer` and a `Fluid3DLayer` on its velocity, with deep_flow's preset (`render.bridge`,
`render.fluid`); PATH=VALUE sets one setting by its panel path under `bridge.` or `fluid.`
(fluid.input.boundary.velocity_drag=50, bridge.readings.lowpass.smooth=300, fluid.input.boundary.velocity_gain=2). The bridge and the
fluid step once per flow frame by the frame's saved interval; `--every N` steps them once per N frames with the
frames' summed interval (the flow still measures every frame), a frame-rate check. `--save DIR` writes every side's
composited output at four steps of the loop, a quarter apart, as 8-bit PNGs (DIR/<side>_<step>.png), for a picture
compared across two runs, and its density seen from the side (DIR/<side>_<step>_side.png, depth across, the front at
the left). A side named `flat` takes no z position: the body goes in at `layer`, as without depth.

Measured per frame on the moving player (the mask, 12 px and more inside, where it moves in the loop):
    body       the boundary velocity's |xy| mean, widths/s: what the water is dragged toward
    water      the fluid velocity's |xy| mean on the injection layer's slice, widths/s
    density    the density's mean (the max of r, g, b) over the whole slice, in density units; 0 for the
               particles, which keep no density volume

Then per side:
    noise      the still part's p99 of the water per pixel (its first 1.5 s left out)
    slow/fast  the mean in each move's slow and fast half, steady (its first 0.3 s left out)
    / body     the fast mean over the body's
    peak       the highest water over the loop against the highest body: a drag at gain 1 stays at 1 or under
    stop       the highest mean from 0.1 s before a move's end to 0.6 s after
    coast      s after a move's end until the water is below 20% of its fast mean
    per width  the density at the loop's end over the body's travel (Σ body × dt), density per width of travel
    GPU        the fluid's step per frame, ms, mean and p95
"""

import json
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import glFinish

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.gl import Texture, Texture3D  # noqa: E402
from modules.flow import Cut, LucasKanadeFlow3D, LucasKanadeFlow3DSettings, TextureKind, VolumeCut  # noqa: E402
from modules.render.layers import (Bridge3DLayer, Bridge3DLayerSettings, BridgeInput, Fluid3DLayer, Fluid3DLayerSettings,  # noqa: E402
                                   FluidInput, Injection)
from bridge_compare import COAST_SHARE, PRESET, SETTLE, coast_s, depth_range, events_max, set_path, windows  # noqa: E402
from depth_velocity_compare import Frames, context, overrides, read_back, recorded_settings, setting_owner, upload  # noqa: E402

STEADY_AFTER: float = 0.3   # s of a half left out of its steady mean


@dataclass
class Series:
    """One measure of one side, per frame: its mean on the moving player, its max there, and the still pixels."""
    player: list[float] = field(default_factory=list)
    peak: list[float] = field(default_factory=list)
    still: list[np.ndarray] = field(default_factory=list)

    def add(self, magnitude: np.ndarray, moving: np.ndarray, core: np.ndarray, still: bool) -> None:
        self.player.append(float(np.mean(magnitude[moving])) if moving.any() else 0.0)
        self.peak.append(float(np.max(magnitude[moving])) if moving.any() else 0.0)
        if still:
            self.still.append(magnitude[core])


class Side:
    """A bridge and a fluid on the flow's velocity, with the preset's settings and the side's own."""

    def __init__(self, spec: str, output: tuple[int, int]) -> None:
        self.name, _, items = spec.partition(':')
        preset = json.loads(PRESET.read_text())['render']
        self.bridge_settings = Bridge3DLayerSettings()
        self.bridge_settings.update_from_dict(preset['bridge'])
        self.fluid_settings = Fluid3DLayerSettings()
        self.fluid_settings.update_from_dict(preset['fluid'])
        self.changed: dict[str, object] = {}
        for item in filter(None, items.split(',')):
            path, value = item.split('=', 1)
            owner, rest = path.split('.', 1)
            settings = {'bridge': self.bridge_settings, 'fluid': self.fluid_settings}.get(owner)
            if settings is None:
                raise SystemExit(f'{path}: a path starts with bridge. or fluid.')
            self.changed[path] = set_path(settings, rest, value)
        self.bridge = Bridge3DLayer(self.bridge_settings)
        self.bridge.allocate(*output)
        self.fluid = Fluid3DLayer(self.fluid_settings)
        self.fluid.allocate(*output)
        self.cut = VolumeCut()
        self.cut.allocate()
        self.series: dict[str, Series] = {'body': Series(), 'water': Series(), 'density': Series()}
        self.milliseconds: list[float] = []

    def step(self, velocity: Texture, mask: Texture, delta_time: float, z_position: Texture | None = None) -> None:
        self.bridge.set_input(BridgeInput(velocity, delta_time, None, mask=mask, z_position=z_position))
        self.bridge.update()
        out = self.bridge.output
        injection = Injection(out.boundary_velocity, out.boundary_density, out.boundary_temperature,
                              out.body_force_acceleration, out.body_force_density, out.body_force_temperature,
                              boundary_velocity_z_position=out.boundary_velocity_z_position,
                              body_force_acceleration_z_position=out.body_force_acceleration_z_position,
                              z_position=out.z_position)
        self.fluid.set_input(FluidInput(delta_time, [injection]))
        glFinish()
        began = time.perf_counter()
        self.fluid.update()
        glFinish()
        self.milliseconds.append((time.perf_counter() - began) * 1000.0)

    def _slice(self, name: str) -> np.ndarray:
        """The fluid's volume `name` on the injection layer, (H, W, 4) at the volume's size."""
        port = self.fluid.ports[name]
        volume = port.texture
        assert isinstance(volume, Texture3D)
        return read_back(self.cut.process(volume, Cut.SLICE, self.fluid_settings.input.layer, TextureKind.SCALAR, port.extent))

    def measure(self, moving: np.ndarray, core: np.ndarray, still: bool) -> None:
        body = read_back(self.bridge.output.boundary_velocity)
        self.series['body'].add(np.hypot(body[..., 0], body[..., 1]), *_fit(moving, core, body.shape[:2]), still)
        water = self._slice('velocity')
        self.series['water'].add(np.hypot(water[..., 0], water[..., 1]), *_fit(moving, core, water.shape[:2]), still)
        if 'density' not in self.fluid.ports:       # the particles keep no density volume
            self.series['density'].player.append(0.0)
            return
        density = self._slice('density')
        amount = np.max(density[..., :3], axis=-1)
        everywhere = np.ones(amount.shape, bool)
        self.series['density'].add(amount, everywhere, everywhere, still)
        self.series['density'].player[-1] = float(np.mean(amount))


def _fit(moving: np.ndarray, core: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The masks at a texture's size, nearest."""
    if moving.shape == shape:
        return moving, core
    size = (shape[1], shape[0])
    return (cv2.resize(moving.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool),
            cv2.resize(core.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool))


def report(side: Side, frames: Frames, w: dict[str, np.ndarray], dt: float) -> list[str]:
    body = np.array(side.series['body'].player)
    water, water_peak = np.array(side.series['water'].player), np.array(side.series['water'].peak)
    body_peak = np.array(side.series['body'].peak)
    s = side.series['water']
    noise = float(np.percentile(np.concatenate(s.still), 99)) if s.still else 0.0
    lines = []
    for name, series in (('body', body), ('water', water)):
        slow, fast = float(np.mean(series[w['slow']])), float(np.mean(series[w['fast']]))
        fast_body = float(np.mean(body[w['fast']]))
        peak = float(np.max(water_peak)) / max(float(np.max(body_peak)), 1e-9) if name == 'water' else 1.0
        line = (f'| {side.name} | {name} | {noise if name == "water" else 0.0:.3f} | {slow:.3f} | {fast:.3f} | '
                f'{fast / max(fast_body, 1e-9):.2f} | {peak:.2f} | {events_max(series, frames, "stop"):.3f} | '
                f'{coast_s(series, frames, fast):.2f} |')
        lines.append(line + (' | |' if name == 'body' else f' {per_width(side, frames):.3f} | '
                             f'{np.mean(side.milliseconds[10:]):.2f} / {np.percentile(side.milliseconds[10:], 95):.2f} |'))
    return lines


def per_width(side: Side, frames: Frames) -> float:
    """The density in the slice at the loop's end over the body's travel, Σ body × dt."""
    body = np.array(side.series['body'].player)
    travel = float(np.sum(body * np.gradient(frames.t))) if len(frames.t) > 1 else 0.0
    return side.series['density'].player[-1] / max(travel, 1e-9)


def save_output(side: Side, path: Path) -> None:
    """The side's composited output as an 8-bit PNG, the color times its alpha, top row first."""
    rgba = read_back(side.fluid.texture)
    rgb = np.clip(rgba[..., :3], 0.0, 1.0)
    cv2.imwrite(str(path), cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if 'density' in side.fluid.ports:      # and the density seen from the side, depth across, front at the left
        port = side.fluid.ports['density']
        volume = port.texture
        assert isinstance(volume, Texture3D)
        cut = read_back(side.cut.process(volume, Cut.SIDE, 0.5, TextureKind.IMAGE, port.extent))
        amount = np.clip(np.max(cut[..., :3], axis=-1), 0.0, 1.0)
        cv2.imwrite(str(path.with_name(path.stem + '_side.png')), (amount * 255.0 + 0.5).astype(np.uint8))


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path)
    parser.add_argument('--side', action='append', default=[], metavar='NAME[:PATH=VALUE,...]')
    parser.add_argument('--every', type=int, default=1, help='step the bridge and the fluid once per N frames')
    parser.add_argument('--save', type=Path, default=None, help='write the outputs at four steps here')
    parser.add_argument('--output', type=int, nargs=2, default=(2160, 3840), metavar=('W', 'H'),
                        help="the fluid's output allocation, as deep_flow's large layers (default 2160 3840)")
    args = parser.parse_args()
    frames = Frames(args.frames)
    width, height = frames.width, frames.height
    context(False, 64, 64)

    flow_settings = LucasKanadeFlow3DSettings()
    flow_settings.update_from_dict(recorded_settings(frames.calls[0]['settings']))
    flow = LucasKanadeFlow3D(flow_settings)
    flow.allocate(width, height)
    sides = [Side(spec, tuple(args.output)) for spec in (args.side or ['preset'])]
    print(frames.describe())
    for side in sides:
        print(f'{side.name}: {side.changed or "as the preset"}')

    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    pending: float = 0.0
    steps = len(frames) // args.every
    saved = {max(1, steps * q // 4) for q in range(1, 5)}
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
    for k in range(len(frames)):
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask.astype(np.float32))
        c = frames.calls[k]
        flow.set_input(luminance)
        flow.set_mask(mask_texture)
        flow.update(depth_texture, c['delta_time'], c['view_span'], tuple(c['depth_step']), depth_range=depth_range())
        pending += c['delta_time']
        if (k + 1) % args.every != 0:
            continue
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        core = mask & (depth > 0) & (inside > 12)
        moving = core & frames.moved
        still = frames.segment(k) == 'still' and frames.t[k] >= frames.still[0] + SETTLE
        for side in sides:
            side.step(flow.velocity, mask_texture, pending, None if side.name == 'flat' else flow.z_position)
            side.measure(moving, core, still)
            step = len(side.milliseconds)
            if args.save is not None and step in saved:
                save_output(side, args.save / f'{side.name}_{step}.png')
        pending = 0.0

    stepped = frames.t[args.every - 1::args.every][:len(sides[0].series['body'].player)]
    frames.t = stepped
    dt = float(np.median(np.diff(stepped))) if len(stepped) > 1 else 1 / 60
    w = windows(frames)
    print(f'{len(stepped)} steps, {dt * 1000:.0f} ms each; coast under {COAST_SHARE:.0%} of the fast mean')
    print('| Side | Measure | Noise | Slow | Fast | / body | Peak / body | Stop | Coast s | Per width | GPU ms mean / p95 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for side in sides:
        print('\n'.join(report(side, frames, w, dt)), flush=True)
    glfw.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
