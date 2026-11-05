"""Play saved frames through the Lucas–Kanade flow and any number of flow bridges, side by side, and measure each.

    python scripts/bridge_compare.py FRAMES --side NAME=KIND[:PATH=VALUE,...] [--side ...] [--flow PATH=VALUE ...]

FRAMES is what depth_track_capture saved from one loop of a recording. No app and no camera: a hidden window
gives the GL context, one `LucasKanadeFlow` measures the saved frames with the settings they were saved with
(`--flow` overrides one by its panel path), and every side's bridge takes its velocity. KIND is `trail` (the
VelocityTrailBridge), `kinetic` (the KineticBridge) or `event` (the VelocityEventBridge: its start and stop);
PATH=VALUE sets one of its settings by its panel path
(trails.fast_smooth, release.stop_gate, onset.force, smooth, ...), the rest as in deep_flow's preset. Each
bridge steps once per flow frame, by the frame's saved interval, as the kinetic bridge does in the app.

What a side gives the fluid, as Motion3DLayer hands it over:
    velocity   its smooth_velocity, in image widths per second (the kinetic one divided by its scale_x)
    amount     the density's alpha: the trail's |velocity| (VelocityMagnitude), the kinetic bridge's emission
    onset      the kinetic bridge's onset_velocity, |xyz| (only with onset.force > 0)
    stop       a trail's max(0, |velocity x/y| - |flow x/y|): how much motion just ended at each pixel

For amount and stop, the last two columns are the burst (s after a move's end until under 20% of that stop's
peak) and the ring's share in the stop windows (4–24 px outside the mask over the body).

Measured on the body (the mask, 12 px and more inside) where it moves in the loop, per frame as a mean, then:
    noise      the still part's p99 per pixel (its first 1.5 s left out, the trails' start)
    slow/fast  the mean in each move's slow and fast half, steady (its first 0.3 s left out)
    start      the highest mean in each move's first 0.4 s
    stop       the highest mean from 0.1 s before a move's end to 0.6 s after
    lag        the delay (ms) that best lines the side's velocity up with the flow's own, over the loop
    coast      s after a move's end until the side's velocity is below 20% of its fast half
    smear      the fast halves' mean |velocity| 4–24 px outside the mask over the one on the body
    GPU        the bridge's time per frame
"""

import json
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import glFinish

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.gl import Texture  # noqa: E402
from modules.flow import (KineticBridge, KineticBridgeSettings, LucasKanadeFlow, LucasKanadeFlowSettings,  # noqa: E402
                          VelocityEventBridge, VelocityEventBridgeSettings, VelocityTrailBridge, VelocityTrailBridgeSettings)
from depth_track_compare import Frames, context, overrides, read_back, recorded_settings, setting_owner, upload  # noqa: E402

SETTLE: float = 1.5         # s of the still part left out of its noise
STEADY_AFTER: float = 0.3   # s of a half left out of its steady mean
START_WINDOW: float = 0.4   # s after a move begins
STOP_WINDOW: tuple[float, float] = (-0.1, 0.6)    # s around a move's end
COAST_SHARE: float = 0.2
MAX_LAG: int = 12           # frames
PRESET: Path = Path(__file__).resolve().parent.parent / 'apps' / 'deep_flow' / 'data' / 'settings' / 'settings.json'


def set_path(settings: object, path: str, value: str) -> object:
    owner, name = setting_owner(settings, path)  # pyright: ignore[reportArgumentType]
    kind = type(getattr(owner, name))
    parsed: object = kind[value.upper()] if issubclass(kind, Enum) else (value.lower() in ('1', 'true', 'on') if kind is bool else kind(float(value)))
    setattr(owner, name, parsed)
    return parsed


@dataclass
class Series:
    """One measure of one side, per frame: its mean on the moving body, outside the mask, and the still pixels."""
    body: list[float] = field(default_factory=list)
    ring: list[float] = field(default_factory=list)
    still: list[np.ndarray] = field(default_factory=list)

    def add(self, magnitude: np.ndarray, moving: np.ndarray, ring: np.ndarray, core: np.ndarray, still: bool) -> None:
        self.body.append(float(np.mean(magnitude[moving])) if moving.any() else 0.0)
        self.ring.append(float(np.mean(magnitude[ring])) if ring.any() else 0.0)
        if still:
            self.still.append(magnitude[core])


class Side:
    def __init__(self, spec: str, width: int, height: int) -> None:
        self.name, rest = spec.split('=', 1)
        kind, _, items = rest.partition(':')
        self.kind = kind
        preset = json.loads(PRESET.read_text())['render']['flow']['flow_bridge']
        self.changed: dict[str, object] = {}
        if kind == 'trail':
            self.settings: VelocityTrailBridgeSettings | KineticBridgeSettings | VelocityEventBridgeSettings = VelocityTrailBridgeSettings()
            self.settings.update_from_dict(preset['velocity_trail'])
            self.bridge: VelocityTrailBridge | KineticBridge | VelocityEventBridge = VelocityTrailBridge(self.settings)
        elif kind == 'kinetic':
            self.settings = KineticBridgeSettings()
            self.settings.update_from_dict(preset['kinetic'])
            self.bridge = KineticBridge(self.settings)
        elif kind == 'event':
            self.settings = VelocityEventBridgeSettings()
            self.settings.update_from_dict(preset.get('velocity_event', {}))
            self.bridge = VelocityEventBridge(self.settings)
        else:
            raise SystemExit(f'unknown bridge {kind}: trail, kinetic or event')
        for item in filter(None, items.split(',')):
            path, value = item.split('=', 1)
            self.changed[path] = set_path(self.settings, path, value)
        self.bridge.allocate(width, height)
        self.series: dict[str, Series] = {'velocity': Series(), 'z': Series(), 'amount': Series(), 'onset': Series(), 'curl': Series(),
                                          'reservoir': Series(), 'start': Series(), 'stop': Series()}
        self.milliseconds: list[float] = []

    def step(self, velocity: Texture, delta_time: float) -> None:
        glFinish()
        began = time.perf_counter()
        self.bridge.set_velocity(velocity)
        self.bridge.update(delta_time)
        if isinstance(self.bridge, KineticBridge):
            self.bridge.update_curl()
        glFinish()
        self.milliseconds.append((time.perf_counter() - began) * 1000.0)

    def measure(self, flow: np.ndarray, moving: np.ndarray, ring: np.ndarray, core: np.ndarray, still: bool) -> None:
        if isinstance(self.bridge, VelocityEventBridge):
            for name, texture in (('start', self.bridge.start), ('stop', self.bridge.stop)):
                self.series[name].add(read_back(texture)[..., 0], moving, ring, core, still)
            return
        v = read_back(self.bridge.smooth_velocity)[..., :3]
        if isinstance(self.bridge, KineticBridge) and isinstance(self.settings, KineticBridgeSettings):
            scale: float = max(abs(self.settings.scale_x), 1e-6)
            v = v / scale
            amount = read_back(self.bridge.emission)[..., 0]
            onset = np.linalg.norm(read_back(self.bridge.onset_velocity)[..., :3], axis=-1) / scale
            curl = np.abs(read_back(self.bridge.curl)[..., 0])
            reservoir = read_back(self.bridge.reservoir)[..., 0]
        else:
            amount = np.linalg.norm(v, axis=-1)
            onset = curl = reservoir = np.zeros(v.shape[:2], np.float32)
        speed: np.ndarray = np.hypot(v[..., 0], v[..., 1])
        stop = (np.maximum(speed - np.hypot(flow[..., 0], flow[..., 1]), 0.0) if self.kind == 'trail'
                else np.zeros(v.shape[:2], np.float32))
        for name, magnitude in (('velocity', speed), ('z', np.abs(v[..., 2])), ('amount', amount),
                                ('onset', onset), ('curl', curl), ('reservoir', reservoir), ('stop', stop)):
            self.series[name].add(magnitude, moving, ring, core, still)


def windows(frames: Frames) -> dict[str, np.ndarray]:
    """Per frame, which window it is in: steady slow, steady fast (per move, all moves), start, stop."""
    t = frames.t
    steady_slow, steady_fast, start, stop = (np.zeros(len(t), bool) for _ in range(4))
    for begin, fast, end in frames.moves:
        steady_slow |= (t >= begin + STEADY_AFTER) & (t < fast)
        steady_fast |= (t >= fast + STEADY_AFTER) & (t < end)
        start |= (t >= begin) & (t < begin + START_WINDOW)
        stop |= (t >= end + STOP_WINDOW[0]) & (t < end + STOP_WINDOW[1])
    return {'slow': steady_slow, 'fast': steady_fast, 'start': start, 'stop': stop}


def events_max(series: np.ndarray, frames: Frames, kind: str) -> float:
    """The mean over the moves of the highest value in each move's start or stop window."""
    t, peaks = frames.t, []
    for begin, _, end in frames.moves:
        window = (t >= begin) & (t < begin + START_WINDOW) if kind == 'start' else (t >= end + STOP_WINDOW[0]) & (t < end + STOP_WINDOW[1])
        if window.any():
            peaks.append(float(np.max(series[window])))
    return float(np.mean(peaks)) if peaks else 0.0


def lag_ms(side: np.ndarray, flow: np.ndarray, dt: float) -> float:
    a, b = flow - flow.mean(), side - side.mean()
    best = max(range(0, MAX_LAG + 1), key=lambda k: float(np.dot(a[:len(a) - k], b[k:])))
    return best * dt * 1000.0


def coast_s(series: np.ndarray, frames: Frames, fast_mean: float) -> float:
    t, coasts = frames.t, []
    for _, _, end in frames.moves:
        after = np.flatnonzero((t >= end) & (series < COAST_SHARE * fast_mean))
        coasts.append(float(t[after[0]] - end) if after.size else float(t[-1] - end))
    return float(np.mean(coasts)) if coasts else 0.0


def burst_s(series: np.ndarray, frames: Frames) -> float:
    """The mean over the moves of the s after a move's end until the series is under 20% of its stop-window peak."""
    t, bursts = frames.t, []
    for _, _, end in frames.moves:
        window = (t >= end + STOP_WINDOW[0]) & (t < end + STOP_WINDOW[1])
        if not window.any():
            continue
        peak_at: int = int(np.flatnonzero(window)[np.argmax(series[window])])
        below = np.flatnonzero((np.arange(len(t)) > peak_at) & (series < COAST_SHARE * series[peak_at]))
        bursts.append(float(t[below[0]] - end) if below.size else float(t[-1] - end))
    return float(np.mean(bursts)) if bursts else 0.0


def report(label: str, series: dict[str, Series], flow: dict[str, Series], frames: Frames, w: dict[str, np.ndarray], dt: float) -> list[str]:
    lines = []
    for name, s in series.items():
        body = np.array(s.body)
        if not body.any():
            continue
        noise = float(np.percentile(np.concatenate(s.still), 99)) if s.still else 0.0
        slow, fast = float(np.mean(body[w['slow']])), float(np.mean(body[w['fast']]))
        start, stop = events_max(body, frames, 'start'), events_max(body, frames, 'stop')
        over_noise: str = f'{fast / noise:.1f}' if noise > 1e-4 else 'no noise'
        line = (f'| {label} | {name} | {noise:.3f} | {slow:.3f} | {fast:.3f} | {fast / max(slow, 1e-9):.2f} | {over_noise} '
                f'| {start:.3f} | {stop:.3f} | {stop / max(fast, 1e-9):.2f} | {start / max(fast, 1e-9):.2f} |')
        if name == 'velocity':
            ring = np.array(s.ring)
            smear = float(np.mean(ring[w['fast']])) / max(fast, 1e-9)
            line += (f' {lag_ms(body, np.array(flow["velocity"].body), dt):.0f} | {coast_s(body, frames, fast):.2f} | {smear:.2f} |')
        elif name in ('stop', 'amount'):
            # A burst: s after a move's end until it is under 20% of that stop's peak; the ring 4-24 px outside the
            # mask over the body, in the stop windows
            ring = np.array(s.ring)
            ring_share = float(np.mean(ring[w['stop']])) / max(float(np.mean(body[w['stop']])), 1e-9)
            line += f' | {burst_s(body, frames):.2f} | {ring_share:.2f} |'
        else:
            line += ' | | |'
        lines.append(line)
    return lines


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path)
    parser.add_argument('--side', action='append', default=[], metavar='NAME=KIND[:PATH=VALUE,...]')
    parser.add_argument('--flow', action='append', default=[], metavar='PATH=VALUE')
    args = parser.parse_args()
    frames = Frames(args.frames)
    width, height = frames.width, frames.height
    context(False, 64, 64)

    flow_settings = LucasKanadeFlowSettings()
    flow_settings.update_from_dict(recorded_settings(frames.calls[0]['settings']))
    for path, value in overrides(args.flow).items():
        owner, name = setting_owner(flow_settings, path)
        setattr(owner, name, value)
    flow = LucasKanadeFlow(flow_settings)
    flow.allocate(width, height)
    sides = [Side(spec, width, height) for spec in args.side]
    print(frames.describe())
    for side in sides:
        print(f'{side.name}: {side.kind} {side.changed or "as the preset"}')

    flow_series: dict[str, Series] = {'velocity': Series(), 'z': Series()}
    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    for k in range(len(frames)):
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask.astype(np.float32))
        c = frames.calls[k]
        flow.set_input(luminance)
        flow.set_mask(mask_texture)
        flow.update(depth_texture, c['delta_time'], c['view_span'], tuple(c['depth_step']))
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        core = mask & (depth > 0) & (inside > 12)
        moving = core & frames.moved
        ring = ~mask & (outside >= 4) & (outside <= 24)
        still = frames.segment(k) == 'still' and frames.t[k] >= frames.still[0] + SETTLE
        v = read_back(flow.velocity)[..., :3]
        flow_series['velocity'].add(np.hypot(v[..., 0], v[..., 1]), moving, ring, core, still)
        flow_series['z'].add(np.abs(v[..., 2]), moving, ring, core, still)
        for side in sides:
            side.step(flow.velocity, c['delta_time'])
            side.measure(v, moving, ring, core, still)

    dt = float(np.median([c['delta_time'] for c in frames.calls]))
    w = windows(frames)
    print('| Side | Measure | Noise | Slow | Fast | Fast / slow | Fast / noise | Start | Stop | Stop / fast | Start / fast | Lag ms | Coast s | Smear |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    print('\n'.join(report('flow', flow_series, flow_series, frames, w, dt)))
    for side in sides:
        print('\n'.join(report(side.name, side.series, flow_series, frames, w, dt)), flush=True)
    print('GPU ms per frame: ' + ', '.join(f'{side.name} {np.median(side.milliseconds):.2f}' for side in sides), flush=True)
    glfw.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
