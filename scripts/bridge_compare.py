"""Play saved frames through the Lucas–Kanade flow and any number of flow bridges, side by side, and measure each.

    python scripts/bridge_compare.py FRAMES --side NAME=KIND[:PATH=VALUE,...] [--side ...] [--flow PATH=VALUE ...]

FRAMES is what depth_velocity_capture saved from one loop of a recording. No app and no camera: a hidden window
gives the GL context, one `LucasKanadeFlow3D` measures the saved frames with the settings they were saved with
(`--flow` overrides one by its panel path), and every side's bridge takes its velocity. KIND is `trail` (the
VelocityTrailBridge) or `change` (the VelocityChangeBridge: its start, bow, stop and wake); PATH=VALUE sets one
of its settings by its panel path (smooth, fast_smooth, stop_region_weight, ...), the rest as in deep_flow's
preset. Each bridge steps once per flow frame, by the frame's saved interval.

What a side gives:
    velocity   a trail's smooth_velocity, in image widths per second
    amount     the density's alpha: the trail's |velocity| (VelocityMagnitude)
    start, bow, stop, wake    an event bridge's events, in image widths per second

For amount and stop, the last two columns are the burst (s after a move's end until under 20% of that stop's
peak) and the ring's share in the stop windows (4–24 px outside the mask over the player).

Measured on the player (the mask, 12 px and more inside) where it moves in the loop, per frame as a mean, then:
    noise      the still part's p99 per pixel (its first 1.5 s left out, the trails' start)
    slow/fast  the mean in each move's slow and fast half, steady (its first 0.3 s left out)
    start      the highest mean in each move's first 0.4 s
    stop       the highest mean from 0.1 s before a move's end to 0.6 s after
    lag        the delay (ms) that best lines the side's velocity up with the flow's own, over the loop
    coast      s after a move's end until the side's velocity is below 20% of its fast half
    smear      the fast halves' mean |velocity| 4–24 px outside the mask over the one on the player
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
from modules.flow import (LucasKanadeFlow3D, LucasKanadeFlow3DSettings,  # noqa: E402
                          Change, VelocityChangeBridge, VelocityChangeBridgeSettings, VelocityTrailBridge, VelocityTrailBridgeSettings,
                          decay_weight)
from depth_velocity_compare import Frames, context, overrides, read_back, recorded_settings, setting_owner, upload  # noqa: E402

SETTLE: float = 1.5         # s of the still part left out of its noise
STEADY_AFTER: float = 0.3   # s of a half left out of its steady mean
START_WINDOW: float = 0.4   # s after a move begins
STOP_WINDOW: tuple[float, float] = (-0.1, 0.6)    # s around a move's end
COAST_SHARE: float = 0.2
MAX_LAG: int = 12           # frames
EPSILON: float = 1e-4       # below this a speed is no motion, as in the bridge
PRESET: Path = Path(__file__).resolve().parent.parent / 'apps' / 'deep_flow' / 'data' / 'settings' / 'settings.json'


def set_path(settings: object, path: str, value: str) -> object:
    owner, name = setting_owner(settings, path)  # pyright: ignore[reportArgumentType]
    kind = type(getattr(owner, name))
    parsed: object = kind[value.upper()] if issubclass(kind, Enum) else (value.lower() in ('1', 'true', 'on') if kind is bool else kind(float(value)))
    setattr(owner, name, parsed)
    return parsed


@dataclass
class Series:
    """One measure of one side, per frame: its mean on the moving player, outside the mask, and the still pixels."""
    player: list[float] = field(default_factory=list)
    ring: list[float] = field(default_factory=list)
    still: list[np.ndarray] = field(default_factory=list)

    def add(self, magnitude: np.ndarray, moving: np.ndarray, ring: np.ndarray, core: np.ndarray, still: bool) -> None:
        self.player.append(float(np.mean(magnitude[moving])) if moving.any() else 0.0)
        self.ring.append(float(np.mean(magnitude[ring])) if ring.any() else 0.0)
        if still:
            self.still.append(magnitude[core])


class Side:
    def __init__(self, spec: str, width: int, height: int) -> None:
        self.name, rest = spec.split('=', 1)
        kind, _, items = rest.partition(':')
        self.kind = kind
        preset = json.loads(PRESET.read_text())['render']['bridge']
        self.changed: dict[str, object] = {}
        if kind == 'trail':
            self.settings: VelocityTrailBridgeSettings | VelocityChangeBridgeSettings = VelocityTrailBridgeSettings()
            self.settings.update_from_dict(preset['readings']['trail'])
            self.bridge: VelocityTrailBridge | VelocityChangeBridge = VelocityTrailBridge(self.settings)
        elif kind == 'change':
            self.settings = VelocityChangeBridgeSettings()
            self.settings.update_from_dict(preset['readings']['change'])
            self.bridge = VelocityChangeBridge(self.settings)
        else:
            raise SystemExit(f'unknown bridge {kind}: trail or change')
        for item in filter(None, items.split(',')):
            path, value = item.split('=', 1)
            self.changed[path] = set_path(self.settings, path, value)
        self.bridge.allocate(width, height)
        self.series: dict[str, Series] = {'velocity': Series(), 'z': Series(), 'amount': Series(),
                                          **{change.name.lower(): Series() for change in Change},
                                          'stop_moving': Series(), 'wake_moving': Series()}
        self.milliseconds: list[float] = []
        # A trial, here only: the stop's region over the flow's moving pixels, as the start's, its two trails per region
        self._region_fast: np.ndarray | None = None
        self._region_slow: np.ndarray | None = None

    def step(self, velocity: Texture, delta_time: float) -> None:
        glFinish()
        began = time.perf_counter()
        self.bridge.set_velocity(velocity)
        self.bridge.update(delta_time)
        glFinish()
        self.milliseconds.append((time.perf_counter() - began) * 1000.0)

    def _moving_stop_share(self, flow: np.ndarray, delta_time: float) -> np.ndarray:
        """The trial stop share per pixel: the region's mean speed over the flow's moving pixels, through the fast and
        the slow trail per region, as far as it falls (numpy; what a moving-pixel stop region would give)."""
        assert isinstance(self.settings, VelocityChangeBridgeSettings)
        s, h, w = self.settings, flow.shape[0], flow.shape[1]
        size: int = max(1, int(round(s.stop_region_size)))
        rows, columns = -(-h // size), -(-w // size)
        speed = np.zeros((rows * size, columns * size))
        speed[:h, :w] = np.linalg.norm(flow, axis=-1)
        blocks = speed.reshape(rows, size, columns, size)
        count = (blocks > EPSILON).sum(axis=(1, 3))
        mean = np.where(count > 0, blocks.sum(axis=(1, 3)) / np.maximum(count, 1), 0.0)
        if self._region_fast is None or self._region_fast.shape != mean.shape:
            self._region_fast, self._region_slow = np.zeros_like(mean), np.zeros_like(mean)
        fast_weight, slow_weight = decay_weight(delta_time, s.fast_smooth), decay_weight(delta_time, s.slow_smooth)
        self._region_fast = fast_weight * self._region_fast + (1.0 - fast_weight) * mean
        self._region_slow = slow_weight * self._region_slow + (1.0 - slow_weight) * mean
        share = np.clip((self._region_slow - self._region_fast) / np.maximum(self._region_slow, EPSILON), 0.0, 1.0)
        share = 1.0 + (share - 1.0) * s.stop_region_weight
        return np.repeat(np.repeat(share, size, axis=0), size, axis=1)[:h, :w]

    def measure(self, flow: np.ndarray, moving: np.ndarray, ring: np.ndarray, core: np.ndarray, still: bool,
                delta_time: float) -> None:
        if isinstance(self.bridge, VelocityChangeBridge):
            values = {change: read_back(self.bridge.change(change))[..., 0] for change in Change}
            for change, value in values.items():
                self.series[change.name.lower()].add(value, moving, ring, core, still)
            fall = values[Change.STOP] + values[Change.WAKE]
            share = self._moving_stop_share(flow, delta_time)
            self.series['stop_moving'].add(fall * share, moving, ring, core, still)
            self.series['wake_moving'].add(fall * (1.0 - share), moving, ring, core, still)
            return
        v = read_back(self.bridge.smooth_velocity)[..., :3]
        for name, magnitude in (('velocity', np.hypot(v[..., 0], v[..., 1])), ('z', np.abs(v[..., 2])),
                                ('amount', np.linalg.norm(v, axis=-1))):
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
        player = np.array(s.player)
        if not player.any():
            continue
        noise = float(np.percentile(np.concatenate(s.still), 99)) if s.still else 0.0
        slow, fast = float(np.mean(player[w['slow']])), float(np.mean(player[w['fast']]))
        start, stop = events_max(player, frames, 'start'), events_max(player, frames, 'stop')
        over_noise: str = f'{fast / noise:.1f}' if noise > 1e-4 else 'no noise'
        line = (f'| {label} | {name} | {noise:.3f} | {slow:.3f} | {fast:.3f} | {fast / max(slow, 1e-9):.2f} | {over_noise} '
                f'| {start:.3f} | {stop:.3f} | {stop / max(fast, 1e-9):.2f} | {start / max(fast, 1e-9):.2f} |')
        if name == 'velocity':
            ring = np.array(s.ring)
            smear = float(np.mean(ring[w['fast']])) / max(fast, 1e-9)
            line += (f' {lag_ms(player, np.array(flow["velocity"].player), dt):.0f} | {coast_s(player, frames, fast):.2f} | {smear:.2f} |')
        elif name in ('amount', 'start', 'bow', 'stop', 'wake', 'stop_moving', 'wake_moving'):
            # A burst: s after a move's end until it is under 20% of that stop's peak; the ring 4-24 px outside the
            # mask over the player, in the stop windows
            ring = np.array(s.ring)
            ring_share = float(np.mean(ring[w['stop']])) / max(float(np.mean(player[w['stop']])), 1e-9)
            line += f' | {burst_s(player, frames):.2f} | {ring_share:.2f} |'
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

    flow_settings = LucasKanadeFlow3DSettings()
    flow_settings.update_from_dict(recorded_settings(frames.calls[0]['settings']))
    for path, value in overrides(args.flow).items():
        owner, name = setting_owner(flow_settings, path)
        setattr(owner, name, value)
    flow = LucasKanadeFlow3D(flow_settings)
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
            side.measure(v, moving, ring, core, still, c['delta_time'])

    dt = float(np.median([c['delta_time'] for c in frames.calls]))
    w = windows(frames)
    print('| Side | Measure | Noise | Slow | Fast | Fast / slow | Fast / noise | Start | Stop | Stop / fast | Start / fast | Lag ms | Coast s | Smear |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    print('\n'.join(report('flow', flow_series, flow_series, frames, w, dt)))
    for side in sides:
        print('\n'.join(report(side.name, side.series, flow_series, frames, w, dt)), flush=True)
    for side in sides:
        if side.kind != 'change':
            continue
        # In steady fast motion every rise and fall is a limb passing: the share its region kept as a start or stop
        # is false; at the real starts and stops the region's share is the right one
        names = ('start', 'bow', 'stop', 'wake', 'stop_moving', 'wake_moving')
        mean = {name: float(np.mean(np.array(side.series[name].player)[w['fast']])) for name in names}
        at_start = {name: events_max(np.array(side.series[name].player), frames, 'start') for name in names}
        at_stop = {name: events_max(np.array(side.series[name].player), frames, 'stop') for name in names}

        def share(values: dict[str, float], kept: str, rest: str) -> float:
            return values[kept] / max(values[kept] + values[rest], 1e-9)
        print(f'{side.name}: steady fast motion read as start {share(mean, "start", "bow"):.2f}, as stop {share(mean, "stop", "wake"):.2f} '
              f'(moving-pixel stop region {share(mean, "stop_moving", "wake_moving"):.2f}); at real starts start '
              f'{share(at_start, "start", "bow"):.2f}; at real stops stop {share(at_stop, "stop", "wake"):.2f} '
              f'(moving-pixel stop region {share(at_stop, "stop_moving", "wake_moving"):.2f})', flush=True)
    print('GPU ms per frame, mean / p95: ' + ', '.join(
        f'{side.name} {np.mean(side.milliseconds[10:]):.2f} / {np.percentile(side.milliseconds[10:], 95):.2f}' for side in sides), flush=True)
    glfw.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
