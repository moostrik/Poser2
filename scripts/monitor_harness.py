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
from OpenGL.GL import (GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_FRAMEBUFFER, GL_QUERY_RESULT, GL_RGBA, GL_RGBA32F,
                       GL_STATIC_DRAW, GL_TEXTURE_2D, GL_TIME_ELAPSED, glBeginQuery, glBindBuffer,
                       glBindFramebuffer, glBindTexture, glBindVertexArray, glBufferData, glEnableVertexAttribArray,
                       glEndQuery, glFinish, glGenBuffers, glGenQueries, glGenVertexArrays, glGetQueryObjectuiv,
                       glTexImage2D, glVertexAttribPointer, glViewport)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.flow import Port, TextureKind  # noqa: E402
from modules.gl import Fbo, Style, Texture  # noqa: E402
from modules.render.layers import FlowStage, MonitorInput, MonitorLayer, MonitorLayerSettings, MonitorView  # noqa: E402

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

VIEWS: tuple[MonitorView, ...] = (MonitorView.IMAGE, MonitorView.COLORMAP, MonitorView.ARROWS,
                                  MonitorView.HISTOGRAM, MonitorView.STRIP)


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
    """The colormap of the middle pixel is 0.5 + 0.5 × the value, as `channel_view.frag` draws it."""
    monitor = Monitor(port, MonitorView.COLORMAP)
    monitor.settings.advanced.legend = False
    monitor.tick(PANEL)
    pixels = monitor.picture()
    h, w = pixels.shape[:2]
    got = pixels[h // 4, w // 4, :3].astype(np.float32) / 255.0                 # a quarter in from the top left
    vh, vw = values.shape[:2]
    want = np.clip(0.5 + 0.5 * values[vh - 1 - vh // 4, vw // 4, :3], 0.0, 1.0)   # the texture's row 0 is its bottom
    results.check('the colormap draws the value around grey', bool(np.all(np.abs(got - want) < 0.02)),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-png', action='store_true', help='skip the pictures')
    args = parser.parse_args()
    window = context()
    results = Results(not args.no_png)
    step_1a(results)
    glFinish()
    glfw.destroy_window(window)
    glfw.terminate()
    print(f"{len(results.failed)} failed: {', '.join(results.failed)}" if results.failed else 'all checks passed')
    if results.png:
        print(f'pictures in {OUT}')
    return 1 if results.failed else 0


if __name__ == '__main__':
    sys.exit(main())
