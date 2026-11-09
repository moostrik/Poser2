"""The monitor harness: the flow monitor's views checked with no app and no camera.

A hidden GL window gives the context. Known textures (and, from step 1b, known volumes) go through
the monitor's views into its FBOs, are read back and checked against numpy, saved as PNGs to look at,
and timed on the GPU. It runs on the GPU like the apps: run it only while no app is running.

    .venv/Scripts/python scripts/monitor_harness.py            # every check, PNGs in temp/monitor_harness
    .venv/Scripts/python scripts/monitor_harness.py --no-png   # the checks and the times only
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from typing import Callable

import cv2
import glfw
import numpy as np
from OpenGL.GL import (GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_FRAMEBUFFER, GL_QUERY_RESULT, GL_R32F, GL_RED, GL_RGBA, GL_RGBA32F,
                       GL_STATIC_DRAW, GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TIME_ELAPSED, glBeginQuery,
                       glBindBuffer, glBindFramebuffer, glBindTexture, glBindVertexArray, glBufferData,
                       glEnableVertexAttribArray, glEndQuery, glFinish, glGenBuffers, glGenQueries, glGenVertexArrays,
                       glGetQueryObjectuiv, glTexImage2D, glTexSubImage3D, glVertexAttribPointer, glViewport)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.flow import REFERENCE_STEP, Cut, OnDemand, Port, TextureKind, VolumeCut  # noqa: E402
from modules.gl import Fbo, MouseButton, Style, Texture, Texture3D  # noqa: E402
from modules.render.layers import Channel, FlowStage, MonitorInput, MonitorLayer, MonitorLayerSettings, MonitorView  # noqa: E402

OUT: str = os.path.join('temp', 'monitor_harness')
PANEL: tuple[int, int] = (540, 960)         # a panel's size, about what a main-window panel gives
OUTPUT: tuple[int, int] = (2160, 3840)      # the layers' allocation size, what the monitor rendered at before


def context() -> object:
    """A hidden GL 4.6 window, current, with the fullscreen quad bound as the app binds it."""
    if not glfw.init():
        raise RuntimeError('GLFW did not start')
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(64, 64, 'monitor harness', None, None)
    if not window:
        raise RuntimeError('no GL window')
    glfw.make_context_current(window)
    vertices = np.array([-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 0, 1], dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindVertexArray(glGenVertexArrays(1))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    Style.reset_state()
    return window


def texture_of(values: np.ndarray) -> Texture:
    """An RGBA32F texture holding `values` (H, W, 4), row 0 at the bottom as GL has it."""
    texture = Texture()
    h, w = values.shape[:2]
    texture.allocate(w, h, GL_RGBA32F)
    glBindTexture(GL_TEXTURE_2D, texture.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, np.ascontiguousarray(values, dtype=np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


def swirl(width: int, height: int) -> np.ndarray:
    """A velocity turning about the middle, x and y up to ±0.8, z a ramp from -0.5 to 0.5 across."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = (x + 0.5) / width - 0.5, (y + 0.5) / height - 0.5
    values = np.zeros((height, width, 4), np.float32)
    values[..., 0], values[..., 1], values[..., 2] = -v * 1.6, u * 1.6, u
    return values


class Monitor:
    """A monitor on one port, drawn into a viewport as the app draws it."""

    def __init__(self, port: Port, view: MonitorView) -> None:
        self.settings = MonitorLayerSettings()
        self.settings.stage = FlowStage.MOTION
        self.settings.texture = 'output.velocity'
        self.settings.view = view
        self.layer = MonitorLayer(self.settings)
        self.layer.allocate(*OUTPUT)
        self.layer.set_input(MonitorInput({FlowStage.MOTION: {'output.velocity': port}}))

    def tick(self, *viewports: tuple[int, int]) -> None:
        """One tick: update, then a draw into each viewport."""
        self.layer.update()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        for width, height in viewports:
            glViewport(0, 0, width, height)
            self.layer.draw()

    def picture(self) -> np.ndarray:
        """What the monitor rendered last, RGBA8 (H, W, 4), top row first."""
        pixels = self.layer.texture.read_to_numpy(flip=True)
        assert pixels is not None
        return pixels


def gpu_ms(work: Callable[[], None], repeats: int = 20) -> float:
    """The GPU time of `work`, the median of `repeats` runs (ms)."""
    query = glGenQueries(1)[0]
    times: list[float] = []
    for _ in range(repeats):
        glBeginQuery(GL_TIME_ELAPSED, query)
        work()
        glEndQuery(GL_TIME_ELAPSED)
        nanoseconds: int = glGetQueryObjectuiv(query, GL_QUERY_RESULT)     # 32 bits hold 4 s, as the app's GpuTimer reads it
        times.append(nanoseconds / 1e6)
    return float(np.median(times))


class Results:
    def __init__(self, png: bool) -> None:
        self.png = png
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = '') -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            self.failed.append(name)

    def save(self, name: str, pixels: np.ndarray) -> None:
        if self.png:
            os.makedirs(OUT, exist_ok=True)
            cv2.imwrite(os.path.join(OUT, f'{name}.png'), cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGRA))


# ── step 1a: the monitor at its drawn size ──────────────────────────────

VIEWS: tuple[MonitorView, ...] = tuple(MonitorView)


def check_drawn_size(results: Results, port: Port) -> None:
    monitor = Monitor(port, MonitorView.COLORMAP)
    monitor.tick(PANEL)
    results.check('renders at its viewport', monitor.layer.texture.width == PANEL[0] and monitor.layer.texture.height == PANEL[1],
                  f'{monitor.layer.texture.width} x {monitor.layer.texture.height}')
    monitor.tick((PANEL[0] // 2, PANEL[1] // 2), PANEL)
    results.check('a larger draw in the tick grows it', monitor.layer.texture.height == PANEL[1])
    monitor.tick((PANEL[0] // 2, PANEL[1] // 2))
    monitor.tick((PANEL[0] // 2, PANEL[1] // 2))
    results.check('a smaller layout takes over a tick later', monitor.layer.texture.height == PANEL[1] // 2)


def check_colormap_values(results: Results, port: Port, values: np.ndarray) -> None:
    """The colormap of a velocity's pixel is its direction's color, 0.5 + 0.5 × the unit vector, times its
    size capped at 1, as `channel_view.frag` draws it."""
    monitor = Monitor(port, MonitorView.COLORMAP)
    monitor.settings.advanced.legend = False
    monitor.tick(PANEL)
    pixels = monitor.picture()
    h, w = pixels.shape[:2]
    got = pixels[h // 4, w // 4, :3].astype(np.float32) / 255.0                 # a quarter in from the top left
    vh, vw = values.shape[:2]
    v = values[vh - 1 - vh // 4, vw // 4, :3]                                   # the texture's row 0 is its bottom
    size = float(np.linalg.norm(v))
    want = np.clip((0.5 + 0.5 * v / max(size, 1e-9)) * min(size, 1.0), 0.0, 1.0)
    results.check('the colormap draws a velocity by its direction, as bright as its size', bool(np.all(np.abs(got - want) < 0.02)),
                  f'got {np.round(got, 3)}, want {np.round(want, 3)}')


def check_views(results: Results, port: Port) -> None:
    for view in VIEWS:
        monitor = Monitor(port, view)
        for _ in range(3):
            monitor.tick(PANEL)
        results.save(f'1a_{view.name.lower()}', monitor.picture())
        legend = monitor.layer._legend.texture
        if view in (MonitorView.HISTOGRAM, MonitorView.STRIP, MonitorView.COLORMAP):
            legend_pixels = legend.read_to_numpy(flip=True)
            if legend_pixels is not None:
                results.save(f'1a_{view.name.lower()}_legend', legend_pixels)
            results.check(f'{view.name.lower()}: the legend at the drawn size', (legend.width, legend.height) == PANEL,
                          f'{legend.width} x {legend.height}')

        def render(monitor: Monitor = monitor, size: tuple[int, int] = PANEL) -> None:
            monitor.tick(size)
        panel_ms = gpu_ms(render)
        for _ in range(2):
            monitor.tick(OUTPUT)                # settle at the output's size, as every monitor rendered before
        output_ms = gpu_ms(lambda: render(size=OUTPUT))
        print(f'     {view.name.lower():<9} {panel_ms:6.2f} ms at {PANEL[0]} x {PANEL[1]}, {output_ms:6.2f} ms at {OUTPUT[0]} x {OUTPUT[1]}')


def step_1a(results: Results) -> None:
    print('step 1a: the monitor at its drawn size')
    values = swirl(512, 288)
    port = Port(texture_of(values), TextureKind.VELOCITY)
    check_drawn_size(results, port)
    check_colormap_values(results, port, values)
    check_views(results, port)


# ── step 1b: volume ports, flat cuts, summaries over a volume ──────────

def volume_of(values: np.ndarray) -> Texture3D:
    """An RGBA32F volume holding `values` (D, H, W, 4): layer 0 the front, row 0 the bottom."""
    volume = Texture3D()
    d, h, w = values.shape[:3]
    volume.allocate(w, h, d, GL_RGBA32F)
    glBindTexture(GL_TEXTURE_3D, volume.tex_id)
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, w, h, d, GL_RGBA, GL_FLOAT, np.ascontiguousarray(values, dtype=np.float32))
    glBindTexture(GL_TEXTURE_3D, 0)
    return volume


def known_volume(width: int, height: int, depth: int) -> np.ndarray:
    """x: the layer from 0 at the front to 1 at the back; y: a sphere in the middle; z: a ramp across
    from -1 to 1; w: a ramp down from 0 at the bottom to 1 at the top."""
    z, y, x = np.mgrid[0:depth, 0:height, 0:width].astype(np.float32)
    u, v, w = (x + 0.5) / width, (y + 0.5) / height, (z + 0.5) / depth
    values = np.zeros((depth, height, width, 4), np.float32)
    values[..., 0] = z / max(depth - 1, 1)
    values[..., 1] = ((u - 0.5) ** 2 + (v - 0.5) ** 2 + (w - 0.5) ** 2 < 0.09).astype(np.float32)
    values[..., 2] = u * 2.0 - 1.0
    values[..., 3] = v
    return values


def read_float(texture: Texture) -> np.ndarray:
    """A float texture's RGBA (H, W, 4), row 0 at the bottom as GL has it."""
    pixels = texture.read_to_numpy(flip=False)
    assert pixels is not None
    return pixels.astype(np.float32)


def check_cuts(results: Results, values: np.ndarray) -> None:
    d, h, w = values.shape[:3]
    volume = volume_of(values)
    cut = VolumeCut()
    cut.allocate()
    extent: float = d / w                        # the top and the side are then a texel per layer

    layer: int = 5
    got = read_float(cut.process(volume, Cut.SLICE, (layer + 0.5) / d, TextureKind.SCALAR, extent))
    results.check('a slice is its layer', bool(np.allclose(got, values[layer], atol=1e-3)))
    got = read_float(cut.process(volume, Cut.FRONT, 0.5, TextureKind.SCALAR, extent))
    results.check('the front mean is the mean through the layers', bool(np.allclose(got, values.mean(axis=0), atol=1e-3)))
    got = read_float(cut.process(volume, Cut.FRONT, 0.5, TextureKind.IMAGE, extent))
    results.check('an image projects its max', bool(np.allclose(got, values.max(axis=0), atol=1e-3)))
    got = read_float(cut.process(volume, Cut.TOP, 0.5, TextureKind.SCALAR, extent))
    results.check('the top is the mean down, the front at the bottom', got.shape[:2] == (d, w)
                  and bool(np.allclose(got, values.mean(axis=1), atol=1e-3)), f'{got.shape[1]} x {got.shape[0]}')
    got = read_float(cut.process(volume, Cut.SIDE, 0.5, TextureKind.SCALAR, extent))
    want = values.mean(axis=2).transpose(1, 0, 2)[:, ::-1]          # (H, D) with the front on the right
    results.check('the side is the mean across, the front on the right', got.shape[:2] == (h, d)
                  and bool(np.allclose(got, want, atol=1e-3)), f'{got.shape[1]} x {got.shape[0]}')
    got = read_float(cut.process(volume, Cut.TOP, 0.5, TextureKind.VELOCITY, extent))
    mean = values.mean(axis=1)
    want = np.stack([mean[..., 0], -mean[..., 2], mean[..., 1], mean[..., 3]], axis=-1)
    results.check("a velocity's top shows x and away from the viewer", bool(np.allclose(got, want, atol=1e-3)))
    cut.deallocate()
    volume.deallocate()


def check_volume_monitor(results: Results, values: np.ndarray) -> None:
    """A volume in the monitor: one refresh a tick for two monitors, every layer counted."""
    d, h, w = values.shape[:3]
    volume = volume_of(values)
    refreshes: list[int] = []
    refresh = OnDemand(lambda: refreshes.append(1))
    port = Port(volume, TextureKind.SCALAR, d / w, refresh)
    a, b = Monitor(port, MonitorView.COLORMAP), Monitor(port, MonitorView.HISTOGRAM)
    for _ in range(3):
        refresh.mark()
        a.tick(PANEL)
        b.tick(PANEL)
    results.check('two monitors on a volume refresh it once a tick', len(refreshes) == 3, f'{len(refreshes)} in 3 ticks')
    results.check('the histogram counts every layer', b.layer._visualizer.histogram_total == w * h * d,
                  f'{b.layer._visualizer.histogram_total}')
    volume.deallocate()


def check_cut_times(results: Results) -> None:
    volume = volume_of(np.zeros((16, 800, 448, 4), np.float32))
    cut = VolumeCut()
    cut.allocate()
    for mode in (Cut.SLICE, Cut.FRONT, Cut.TOP, Cut.SIDE):
        ms = gpu_ms(lambda mode=mode: cut.process(volume, mode, 0.5, TextureKind.SCALAR, 0.14), 10)
        print(f'     {mode.name.lower():<9} {ms:6.2f} ms of a 448 x 800 x 16 volume')
    cut.deallocate()
    volume.deallocate()


def step_1b(results: Results) -> None:
    print('step 1b: volume ports, flat cuts, summaries over a volume')
    values = known_volume(64, 48, 8)
    check_cuts(results, values)
    check_volume_monitor(results, values)
    check_cut_times(results)


# ── step 3: the cube ──────────────────────────────────────────────────────

def cube_monitor(values: np.ndarray, kind: TextureKind, view: MonitorView, extent: float = 0.5) -> Monitor:
    monitor = Monitor(Port(volume_of(values), kind, extent), view)
    monitor.settings.advanced.legend = False
    return monitor


def middle(monitor: Monitor) -> np.ndarray:
    return monitor.picture()[PANEL[1] // 2, PANEL[0] // 2].astype(np.float32)


def check_cube_order(results: Results) -> None:
    """The near side covers the far one from both sides: the front's red from the front, the back's blue from behind."""
    d, h, w = 8, 48, 64
    layered = np.zeros((d, h, w, 4), np.float32)
    layered[:2, ..., 0] = 1.0                    # the front red
    layered[-2:, ..., 2] = 1.0                   # the back blue
    monitor = cube_monitor(layered, TextureKind.IMAGE, MonitorView.COLORMAP)
    for yaw_drag, name in ((0.0, 'front'), (-1.0, 'back')):
        monitor.layer.reset_view()
        monitor.layer.drag(yaw_drag, 0.0, MouseButton.LEFT)
        monitor.tick(PANEL)
        red, blue = middle(monitor)[0], middle(monitor)[2]
        results.check(f'the cube from the {name} shows its near side over its far side', (red > blue) == (name == 'front'),
                      f'red {red:.0f}, blue {blue:.0f}')
        results.save(f'3_cube_layers_{name}', monitor.picture())

    monitor.layer.reset_view()
    monitor.settings.advanced.slice = True
    monitor.settings.advanced.slice_focus = 0.9     # the slice well in focus, the front at a tenth
    monitor.layer.scroll(d, d)                   # the back layer
    monitor.tick(PANEL)
    red, blue = middle(monitor)[0], middle(monitor)[2]
    results.check('the slice on the back layer brings it forward of the dimmed front', blue > red, f'red {red:.0f}, blue {blue:.0f}')
    results.save('3_cube_slice_back', monitor.picture())


def check_cube_colors(results: Results) -> None:
    """A velocity by its direction alone: slow and fast +x both reddish, the empty volume clear."""
    d, h, w = 8, 48, 64
    for speed in (0.1, 1.0):
        flow = np.zeros((d, h, w, 4), np.float32)
        flow[..., 0] = speed
        monitor = cube_monitor(flow, TextureKind.VELOCITY, MonitorView.COLORMAP)
        monitor.settings.advanced.scale = 1.0 / speed        # full either way: the color alone is compared
        monitor.tick(PANEL)
        r, g, b, _ = middle(monitor)
        results.check(f'a velocity of {speed} along +x is colored by its direction', r > g + 40 and abs(g - b) < 10,
                      f'rgb {r:.0f} {g:.0f} {b:.0f}')
    empty = cube_monitor(np.zeros((d, h, w, 4), np.float32), TextureKind.SCALAR, MonitorView.COLORMAP)
    empty.tick(PANEL)
    results.check('an empty volume draws nothing, only the edges', int(middle(empty)[3]) == 0, f'alpha {middle(empty)[3]}')


def check_cube_sphere(results: Results) -> None:
    d, h, w = 8, 48, 64
    sphere = known_volume(w, h, d)
    sphere[..., 0] = sphere[..., 1]
    cube = cube_monitor(sphere, TextureKind.SCALAR, MonitorView.COLORMAP)
    cube.settings.advanced.channel = Channel.X
    cube.layer.drag(0.15, 0.1, MouseButton.LEFT)
    cube.tick(PANEL)
    pixels = cube.picture()
    alpha = pixels[..., 3].astype(np.float32)
    ys, xs = np.nonzero(alpha > 0.3 * alpha.max())
    centered = len(xs) > 0 and abs(float(xs.mean()) - PANEL[0] / 2) < PANEL[0] * 0.1 and abs(float(ys.mean()) - PANEL[1] / 2) < PANEL[1] * 0.1
    results.check("a sphere's cube is densest at the box's middle", centered, f'{len(xs)} pixels')
    results.save('3_cube_sphere', pixels)
    cube.settings.advanced.slice = True
    cube.tick(PANEL)
    results.save('3_cube_sphere_slice', cube.picture())
    results.check('the cube renders at the drawn size', (cube.layer.texture.width, cube.layer.texture.height) == PANEL)


def check_cube_times(results: Results) -> None:
    big = cube_monitor(known_volume(448, 800, 16), TextureKind.SCALAR, MonitorView.COLORMAP, 0.14)
    for slice_on in (False, True):
        big.settings.advanced.slice = slice_on
        big.tick(PANEL)
        ms = gpu_ms(lambda: big.tick(PANEL), 10)
        print(f'     the cube{" + slice" if slice_on else ""}  {ms:6.2f} ms of a 448 x 800 x 16 volume at {PANEL[0]} x {PANEL[1]}')


def step_3(results: Results) -> None:
    print('step 3: the cube')
    check_cube_order(results)
    check_cube_colors(results)
    check_cube_sphere(results)
    check_cube_times(results)


# ── step 4: glyphs ────────────────────────────────────────────────────────

def drawn(pixels: np.ndarray) -> np.ndarray:
    """The pixels something was drawn into (alpha above zero)."""
    return pixels[pixels[..., 3] > 0]


def check_flat_dots(results: Results) -> None:
    """A scalar's dot in its cell: the colormap's color at its center."""
    values = np.zeros((288, 512, 4), np.float32)
    values[..., 0] = 0.8
    monitor = Monitor(Port(texture_of(values), TextureKind.SCALAR), MonitorView.GLYPHS)
    monitor.settings.advanced.channel = Channel.X
    monitor.settings.advanced.legend = False
    monitor.tick(PANEL)
    pixels = monitor.picture()
    spacing: int = int(monitor.settings.advanced.glyph_spacing.value)
    center = pixels[spacing // 2, spacing // 2, :3].astype(np.float32) / 255.0      # the first cell's middle, from the top left
    results.check("a scalar's dot has the colormap's color", bool(np.allclose(center, (0.9, 0.5, 0.5), atol=0.03)),
                  f'{np.round(center, 2)}')
    results.save('4_dots_flat', pixels)

    one = Texture()                                         # a one-channel texture marked a velocity: dots, not diagonal arrows
    one.allocate(512, 288, GL_R32F)
    glBindTexture(GL_TEXTURE_2D, one.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 512, 288, 0, GL_RED, GL_FLOAT, np.full((288, 512), 0.8, np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    monitor = Monitor(Port(one, TextureKind.VELOCITY), MonitorView.GLYPHS)
    monitor.settings.advanced.legend = False
    monitor.tick(PANEL)
    pixels = monitor.picture()
    center = pixels[spacing // 2, spacing // 2, :3].astype(np.float32) / 255.0
    results.check('a one-channel texture draws dots, whatever its kind', bool(np.allclose(center, (0.9, 0.9, 0.9), atol=0.05)),
                  f'{np.round(center, 2)}')
    results.save('4_dots_one_channel', pixels)

    push = np.zeros((288, 512, 4), np.float32)
    push[..., 0] = 0.8 / REFERENCE_STEP                     # 0.8 widths/s per step along +x
    arrows, pushes = (Monitor(Port(texture_of(push * (REFERENCE_STEP if kind == TextureKind.VELOCITY else 1.0)), kind), MonitorView.GLYPHS)
                      for kind in (TextureKind.VELOCITY, TextureKind.ACCELERATION))
    for monitor in (arrows, pushes):
        monitor.settings.advanced.legend = False
        monitor.tick(PANEL)
    filled = len(drawn(pushes.picture())) > len(drawn(arrows.picture()))
    results.check("an acceleration draws a filled push, the arrow's length per step", filled,
                  f'{len(drawn(pushes.picture()))} against {len(drawn(arrows.picture()))} pixels')
    results.save('4_pushes_flat', pushes.picture())


def glyph_monitor(values: np.ndarray, kind: TextureKind, extent: float = 0.5) -> Monitor:
    monitor = Monitor(Port(volume_of(values), kind, extent), MonitorView.GLYPHS)
    monitor.settings.advanced.legend = False
    return monitor


def check_3d_glyphs(results: Results) -> None:
    d, h, w = 8, 48, 64
    toward = np.zeros((d, h, w, 4), np.float32)
    toward[..., 2] = 1.0                                     # every arrow toward the camera of the front view
    monitor = glyph_monitor(toward, TextureKind.VELOCITY)
    monitor.tick(PANEL)
    pixels = drawn(monitor.picture())
    blue = len(pixels) > 0 and float(pixels[:, 2].mean()) > float(pixels[:, 0].mean())
    results.check('an arrow pointing at the camera still shows, in its direction color', blue, f'{len(pixels)} pixels')
    results.save('4_arrows_toward', monitor.picture())

    faint = np.zeros((d, h, w, 4), np.float32)
    faint[..., 0] = 0.02                                     # far below a tenth of full: a dot, still +x's color
    monitor = glyph_monitor(faint, TextureKind.VELOCITY)
    monitor.tick(PANEL)
    pixels = drawn(monitor.picture())
    reddish = len(pixels) > 0 and float(pixels[:, 0].mean()) > float(pixels[:, 1].mean()) + 20
    results.check('a tiny force draws a dot in its direction color', reddish, f'{len(pixels)} pixels')
    results.save('4_arrows_tiny', monitor.picture())

    swirl_volume = np.repeat(swirl(w, h)[None], d, axis=0)
    swirl_volume[..., 2] = np.linspace(-1.0, 1.0, d)[:, None, None]      # z from the back to the front
    monitor = glyph_monitor(swirl_volume, TextureKind.VELOCITY)
    for name, drags in (('front', []), ('turned', [(0.2, 0.15)])):
        monitor.layer.reset_view()
        for dx, dy in drags:
            monitor.layer.drag(dx, dy, MouseButton.LEFT)
        monitor.tick(PANEL)
        results.save(f'4_arrows_{name}', monitor.picture())

    sphere = known_volume(w, h, d)
    sphere[..., 0] = sphere[..., 1]                          # the dots' size is x: the sphere
    dots = glyph_monitor(sphere, TextureKind.SCALAR)
    dots.layer.drag(0.15, 0.1, MouseButton.LEFT)
    dots.tick(PANEL)
    results.check('a scalar volume draws its dots', len(drawn(dots.picture())) > 0)
    results.save('4_dots_turned', dots.picture())
    empty = glyph_monitor(np.zeros((d, h, w, 4), np.float32), TextureKind.SCALAR)
    empty.tick(PANEL)
    drawn_pixels = drawn(empty.picture())
    results.check('values too small to see draw no dots; without the overlay, no outlines either', len(drawn_pixels) == 0,
                  f'{len(drawn_pixels)} pixels')

    big = glyph_monitor(np.repeat(np.repeat(np.repeat(swirl(448, 800)[None], 16, axis=0), 1, axis=1), 1, axis=2), TextureKind.VELOCITY, 0.14)
    big.tick(PANEL)
    ms = gpu_ms(lambda: big.tick(PANEL), 10)
    print(f'     3D arrows  {ms:6.2f} ms of a 448 x 800 x 16 volume at {PANEL[0]} x {PANEL[1]}, spacing 32')


def step_4(results: Results) -> None:
    print('step 4: glyphs')
    check_flat_dots(results)
    check_3d_glyphs(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-png', action='store_true', help='skip the pictures')
    args = parser.parse_args()
    window = context()
    results = Results(not args.no_png)
    step_1a(results)
    step_1b(results)
    step_3(results)
    step_4(results)
    glFinish()
    glfw.destroy_window(window)
    glfw.terminate()
    print(f"{len(results.failed)} failed: {', '.join(results.failed)}" if results.failed else 'all checks passed')
    if results.png:
        print(f'pictures in {OUT}')
    return 1 if results.failed else 0


if __name__ == '__main__':
    sys.exit(main())
