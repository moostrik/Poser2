"""The shader check: every shader of `modules/gl` drawn once on known input and held to a reference image.

A hidden GL 4.6 window gives the context; it is this script's own main thread. Each case draws into a 128 x 128
RGBA8 FBO with blending off, cleared to transparent black, is read back and compared with the reference its first run
made, within 1/255 per channel: a GPU's output is not bit-exact across two code paths that draw the same. Shaders
with no draw to compare (a compute shader) are compiled only. It runs on the GPU like the apps: run it only while no
app is running.

    .venv/Scripts/python scripts/shader_check.py --reference   # make the reference images, temp/shader_check
    .venv/Scripts/python scripts/shader_check.py               # compare; PNGs of each case beside the reference
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections.abc import Callable
from typing import Protocol

import cv2
import glfw
import numpy as np
from OpenGL.GL import (GL_FLOAT, GL_R8, GL_R32F, GL_RED, GL_RGBA, GL_RGBA8, GL_TEXTURE_2D, GL_UNPACK_ALIGNMENT,
                       GL_UNSIGNED_BYTE, glBindTexture, glFinish, glPixelStorei, glTexSubImage2D, glViewport)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apps.hd_trio.render.layers.center.shaders import CelShade  # noqa: E402
from apps.smile.render.shaders import FaceColor, FlashIn, FlashOut, Omission  # noqa: E402
from modules.flow import DepthCrop, DepthUpscale  # noqa: E402
from modules.flow.shaders import AddRegion, ChannelAddRegion, ChannelSetRegion, SetRegion  # noqa: E402
from modules.flow.visualization.shaders import ColormapKey  # noqa: E402
from modules.gl import Bloom, Colormap, Context, DistanceField, Fbo, Style, Text, Texture, shaders  # noqa: E402
from modules.gl.shaders.filter import KERNEL_SIGMA  # noqa: E402
from modules.pose import PointLandmark, Points2D  # noqa: E402
from modules.inference.preprocessing.shaders import DepthAlign  # noqa: E402
from modules.render import PosePointLines  # noqa: E402
from modules.render.modules.board.shaders import TimeSeries  # noqa: E402
from modules.render.modules.pose.shaders import AddMix, Feature, LayerComposite, Window  # noqa: E402
from modules.render.modules.flow.shaders import DepthColormap, DepthDrawMode  # noqa: E402
from modules.render.general.screen_recording.shaders import RgbToYuv, YuvPlane  # noqa: E402
from modules.utils import Color, Point2f, Quad, Rect  # noqa: E402

OUT: str = os.path.join('temp', 'shader_check')
SIZE: int = 128
TOLERANCE: int = 1                  # per channel, of 255
LUT: str = os.path.join('data', 'luts', 'Blockbuster.cube')


def context() -> glfw._GLFWwindowPointerT:
    """A hidden GL 4.6 window, current, with its `gl.Context`."""
    if not glfw.init():
        raise RuntimeError('GLFW did not start')
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(64, 64, 'shader check', None, None)
    if not window:
        raise RuntimeError('no GL window')
    glfw.make_context_current(window)
    Context().allocate()
    return window


def texture(pixels: np.ndarray) -> Texture:
    """A texture holding `pixels` (uint8, H x W x 4 or H x W), row 0 at GL's bottom."""
    tex = Texture()
    four: bool = pixels.ndim == 3
    tex.allocate(pixels.shape[1], pixels.shape[0], GL_RGBA8 if four else GL_R8)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, tex.tex_id)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pixels.shape[1], pixels.shape[0], GL_RGBA if four else GL_RED,
                    GL_UNSIGNED_BYTE, np.ascontiguousarray(pixels))
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def depth_texture(meters: np.ndarray) -> Texture:
    """A one-channel float texture holding `meters`, row 0 at GL's bottom."""
    tex = Texture()
    tex.allocate(meters.shape[1], meters.shape[0], GL_R32F)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, tex.tex_id)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, meters.shape[1], meters.shape[0], GL_RED, GL_FLOAT,
                    np.ascontiguousarray(meters, dtype=np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def inputs() -> dict[str, Texture]:
    """The known inputs: a color gradient, a second image, a soft round mask, a hard-edged one, and a depth."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / (SIZE - 1)
    gradient = np.dstack([x, y, 1.0 - x, 0.5 + 0.5 * x]) * 255
    checker = ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(np.float32)
    second = np.dstack([checker, 0.5 * checker + 0.25, 1.0 - checker, np.ones_like(x)]) * 255
    radius = np.hypot(x - 0.5, y - 0.5)
    soft = np.clip(1.0 - radius * 2.5, 0.0, 1.0) * 255
    hard = (radius < 0.3).astype(np.float32) * 255
    depth = 0.2 + 0.5 * x + 0.2 * (radius < 0.25)      # a slope with a step, under 1 so the RGBA8 target holds it
    return {'gradient': texture(gradient.astype(np.uint8)), 'second': texture(second.astype(np.uint8)),
            'soft': texture(soft.astype(np.uint8)), 'hard': texture(hard.astype(np.uint8)),
            'depth': depth_texture(depth)}


def pose() -> Points2D:
    """A standing figure, every point found."""
    values = np.full((len(PointLandmark), 2), 0.5, dtype=np.float32)
    for i, landmark in enumerate(PointLandmark):
        values[i] = (0.3 + 0.4 * ((i * 7) % 5) / 4, 0.1 + 0.8 * i / (len(PointLandmark) - 1))
    return Points2D(values, np.ones(len(PointLandmark), dtype=np.float32))


Case = Callable[[dict[str, Texture]], None]


class Allocated(Protocol):
    """What a case allocates before it draws."""
    def allocate(self) -> object: ...


class HarnessFeature:
    """A feature of eight known values and scores, as `Feature` reads one."""
    values: np.ndarray = np.linspace(-0.8, 0.9, 8).astype(np.float32)
    scores: np.ndarray = np.linspace(0.2, 1.0, 8).astype(np.float32)

    def __len__(self) -> int:
        return len(self.values)

    def display_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)


def exact_distance(field: DistanceField) -> None:
    """Jump flooding held to the exact distance: every pixel's distance to the nearest mask pixel, by brute force."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / (SIZE - 1)
    mask = (np.hypot(x - 0.5, y - 0.5) < 0.3)                         # the 'hard' input, as `inputs` makes it
    seeds = np.argwhere(mask).astype(np.float32)
    pixels = np.indices((SIZE, SIZE)).reshape(2, -1).T.astype(np.float32)
    exact = np.sqrt(((pixels[:, None, :] - seeds[None, :, :]) ** 2).sum(-1).min(1)).reshape(SIZE, SIZE)
    measured = field.texture.read_to_numpy()
    assert measured is not None
    measured = np.asarray(measured, dtype=np.float32).reshape(SIZE, SIZE, -1)[..., 0]
    error = np.abs(measured - exact)
    print(f'{"distance_field exact":24s} {"ok" if error.max() <= 1.0 else "DIFFERS"}  max {error.max():.3f} px, '
          f'{(error > 1e-3).mean() * 100:.2f}% of pixels off')


def cases() -> dict[str, tuple[Allocated | None, Case | None, Case | None]]:
    """Each case: the shader it allocates (None for one that is not a shader), its draw, and what it draws into its
    own targets before (an effect over a chain of passes)."""
    s = shaders
    c: dict[str, tuple[Allocated | None, Case | None, Case | None]] = {}

    def add(name: str, shader: Allocated | None, draw: Case | None, prepare: Case | None = None) -> None:
        c[name] = (shader, draw, prepare)

    # The texture's draws, through the context's Blit (no shader of their own to allocate)
    add('blit', None, lambda t: t['gradient'].draw())
    add('blit_opacity', None, lambda t: t['gradient'].draw(opacity=0.5))
    add('blit_flip_y', None, lambda t: t['gradient'].draw(src=Rect(0.0, 1.0, 1.0, -1.0)))
    add('blit_flip_x', None, lambda t: t['gradient'].draw(src=Rect(1.0, 0.0, -1.0, 1.0)))
    add('blit_region', None, lambda t: t['gradient'].draw(src=Rect(0.25, 0.25, 0.5, 0.5)))
    add('draw_dst', None, lambda t: t['gradient'].draw(dst=Rect(0.1, 0.2, 0.5, 0.4)))
    add('draw_dst_src', None, lambda t: t['second'].draw(Rect(0.5, 0.0, 0.5, 1.0), Rect(0.0, 0.5, 0.5, 0.5)))

    contrast = s.BrightnessContrast()
    add('brightness_contrast', contrast, lambda t: contrast.use(t['gradient'], 0.1, 1.5))
    exposure = s.Exposure()
    add('exposure', exposure, lambda t: exposure.use(t['gradient'], 1.5, 0.05, 0.8))
    hsl = s.Hsl()
    add('hsl', hsl, lambda t: hsl.use(t['gradient'], 0.2, 1.3, 0.1))
    hsv = s.Hsv()
    add('hsv', hsv, lambda t: hsv.use(t['gradient'], 0.2, 1.3, 1.1))
    lut = s.Lut()
    add('lut', lut, lambda t: lut.use(t['gradient'], 0.8))
    noise = s.WhiteNoise()

    def draw_noise(_: dict[str, Texture]) -> None:
        random.seed(0)
        noise.use()
    add('white_noise', noise, draw_noise)
    simplex = s.SimplexNoise()

    add('simplex_noise', simplex, lambda t: simplex.use(12.5, 1.0, 0.5))
    rgb_to_yuv = RgbToYuv()
    add('rgb_to_yuv_y', rgb_to_yuv, lambda t: rgb_to_yuv.use(t['gradient'], YuvPlane.Y))
    add('rgb_to_yuv_cbcr', rgb_to_yuv, lambda t: rgb_to_yuv.use(t['gradient'], YuvPlane.CBCR))
    sharpen = s.Sharpen()
    add('sharpen', sharpen, lambda t: sharpen.use(t['second'], 1.5))

    blur = s.Blur()
    add('blur_across', blur, lambda t: blur.use(t['second'], True, 3.0 * KERNEL_SIGMA))
    add('blur_down', blur, lambda t: blur.use(t['second'], False, 3.0 * KERNEL_SIGMA))
    wipe = s.BarnDoorWipe()
    add('barn_door_wipe', wipe, lambda t: wipe.use(t['gradient'], t['second'], 0.4))
    flash_in = FlashIn()
    add('flash_in', flash_in, lambda t: flash_in.use(t['gradient'], t['second'], 0.7))
    flash_out = FlashOut()
    add('flash_out', flash_out, lambda t: flash_out.use(t['gradient'], t['second'], t['soft'], 0.6))
    gradient_wipe = s.GradientWipe()
    add('gradient_wipe', gradient_wipe, lambda t: gradient_wipe.use(t['gradient'], t['second'], t['soft'], 0.5, 0.2))
    add('gradient_wipe_hard', gradient_wipe, lambda t: gradient_wipe.use(t['gradient'], t['second'], t['soft'], 0.5))
    omission = Omission()
    add('omission', omission, lambda t: omission.use(t['gradient'], t['second'], t['soft'], 0.3, 0.6))

    mix = s.Mix()
    add('mix', mix, lambda t: mix.use(t['gradient'], t['second'], 0.3))
    color_fill = s.ColorFill()
    add('color_fill', color_fill, lambda t: color_fill.use(t['soft'], Color(0.9, 0.4, 0.1, 1.0)))
    add('draw_roi_turned_wide', None,
        lambda t: t['gradient'].draw(src=Rect(0.2, 0.3, 0.5, 0.6).turned(0.7, Point2f(0.4, 0.55), 1.6)))
    add('draw_roi_turned', None,
        lambda t: t['gradient'].draw(src=Rect(0.2, 0.3, 0.5, 0.6).turned(-0.4, Point2f(0.45, 0.6), 1.0)))
    circles = s.FillCircles()
    add('fill_circles', circles,
        lambda t: circles.use([(0.25, 0.75), (0.6, 0.3), (0.8, 0.7)], 0.08, 0.01, Color(1.0, 0.5, 0.2, 1.0), 1.0))
    rect = s.FillRect()
    add('fill_rect', rect, lambda t: rect.use(Rect(0.1, 0.2, 0.5, 0.3), Color(0.2, 0.8, 0.4, 0.9)))
    outline = s.StrokeRect()
    add('stroke_rect', outline,
        lambda t: outline.use(Rect(0.1, 0.2, 0.5, 0.3), Color(1.0, 1.0, 1.0, 1.0), 0.02, 0.03))

    mask_apply = s.MaskApply()
    add('mask_apply', mask_apply, lambda t: mask_apply.use(t['gradient'], t['soft'], 1.0))
    dilate, erode = s.Dilate(), s.Erode()
    add('dilate', dilate, lambda t: dilate.use(t['hard'], 3.0))
    add('erode', erode, lambda t: erode.use(t['hard'], 3.0))
    threshold = s.Threshold()
    add('threshold', threshold, lambda t: threshold.use(t['soft'], 0.4))
    sobel = s.Sobel()
    add('sobel', sobel, lambda t: sobel.use(t['second'], 0.1, 1.0, False))
    adaptive = s.AdaptiveThreshold()
    add('adaptive_threshold', adaptive, lambda t: adaptive.use(t['second'], 0.5, 1.0, 3.0, False))

    lines = PosePointLines()
    figure = pose()
    add('pose_point_lines', lines, lambda t: lines.use(figure, 0.02, 0.005, (1.0, 0.8, 0.2, 1.0), True))

    text = Text()

    def draw_text(_: dict[str, Texture]) -> None:
        text.draw_text(4, 8, 'Poser 0123', Color(1.0, 1.0, 1.0, 1.0), SIZE, SIZE)
        text.draw_box_text(4, 60, 'box 45', Color(1.0, 0.9, 0.2, 1.0), Color(0.0, 0.0, 0.3, 0.7), SIZE, SIZE)
    add('text', text, draw_text)

    # The shaders outside the GL that take a position (the GL plan's H)
    face = FaceColor()
    fields = [(0.1, 0.1, 0.3, 0.3, 0.8), (0.55, 0.5, 0.3, 0.35, 0.4)]
    add('face_color', face, lambda t: face.use(t['gradient'], SIZE, SIZE, fields, 0, 0.8, (0.2, 0.15, 0.5, 0.6, 0.3),
                                               0.5, 1, (1.0, 0.6, 0.1, 1.0)))
    set_region, add_region = SetRegion(), AddRegion()
    add('set_region', set_region, lambda t: set_region.use(t['gradient'], t['second'], 0.2, 0.5, 0.5, 0.4, 0.8))
    add('add_region', add_region, lambda t: add_region.use(t['gradient'], t['second'], 0.2, 0.5, 0.5, 0.4, 0.5))
    channel_set, channel_add = ChannelSetRegion(), ChannelAddRegion()
    add('channel_set_region', channel_set,
        lambda t: channel_set.use(t['gradient'], t['soft'], 1, 0.2, 0.5, 0.5, 0.4, 0.8))
    add('channel_add_region', channel_add,
        lambda t: channel_add.use(t['gradient'], t['soft'], 2, 0.2, 0.5, 0.5, 0.4, 0.5))
    upscale = DepthUpscale()
    add('depth_upscale', upscale, lambda t: upscale.use(t['depth'], Rect(0.2, 0.3, 0.5, 0.6), 0.1, t['soft'], 0.5, 2.0))
    crop = DepthCrop()
    add('depth_crop', crop, lambda t: crop.use(t['depth'], Rect(0.2, 0.3, 0.5, 0.6), 0.5, 2.0))
    key = ColormapKey()
    add('colormap_key', key, lambda t: key.use((10.0 / SIZE, 38.0 / SIZE, 100.0 / SIZE, 70.0 / SIZE), 4, 3,
                                               12.0 / SIZE, 6.0 / SIZE))

    # The shaders outside the GL that the GL plan's lines M to P change (L3)
    add('trail_blur', blur, lambda t: blur.use(t['second'], True, 1.7))       # the 2D trail's blur (N6)
    weighted_sum = s.WeightedSum()      # the flow's Add, Set and Trail before (P5)
    add('flow_add', weighted_sum, lambda t: weighted_sum.use(t['gradient'], 0.6, t['second'], 0.3))
    add('flow_set', weighted_sum, lambda t: weighted_sum.use(t['gradient'], 0.7))
    add('flow_trail', weighted_sum, lambda t: weighted_sum.use(t['gradient'], 0.6, t['second'], 0.3))
    guided = s.GuidedFilter()
    add('guided_products', guided, lambda t: guided.products(t['gradient'], t['hard']))
    add('guided_box', guided, lambda t: guided.box(t['gradient'], True, 3))
    add('guided_coefficients', guided, lambda t: guided.coefficients(t['gradient'], 0.01))
    add('guided_output', guided, lambda t: guided.output(t['gradient'], t['second']))
    mask_multiply = s.MaskMultiply()
    add('mask_multiply', mask_multiply, lambda t: mask_multiply.use(t['gradient'], t['soft']))
    mask_aa = s.MaskAntialias()
    add('mask_antialias', mask_aa, lambda t: mask_aa.use(t['soft']))
    colorize_shader = s.Colorize()
    add('colorize', colorize_shader, lambda t: colorize_shader.use(t['gradient'], Color(0.9, 0.3, 0.1), 0.6))
    cel = CelShade()
    add('cel_shade', cel, lambda t: cel.use(t['gradient'], 1.2, 1.0, 0.0, 1.1, 5, 0.2, 1.5))
    depth_view = DepthColormap()
    for mode in range(3):
        add(f'depth_view_{mode}', depth_view, lambda t, m=mode: depth_view.use(t['depth'], t['gradient'], 0.3, 0.8, DepthDrawMode(m)))
    for colormap in (Colormap.TURBO, Colormap.VIRIDIS):
        add(f'depth_view_{colormap.name.lower()}', depth_view,
            lambda t, c=colormap: depth_view.use(t['depth'], t['gradient'], 0.3, 0.8, DepthDrawMode.COLOR, c))

    quarter_turn = Quad(Point2f(1.0, 0.0), Point2f(1.0, 1.0), Point2f(0.0, 1.0), Point2f(0.0, 0.0))
    add('quarter_turn', None, lambda t: t['gradient'].draw(src=quarter_turn))

    # The standard shaders the GL plan's line R adds
    tone_map = s.ToneMap()
    for operator in s.ToneMapOperator:
        add(f'tone_map_{operator.name.lower()}', tone_map, lambda t, o=operator: tone_map.use(t['gradient'], o, 3.0))
    levels = s.Levels()
    add('levels', levels, lambda t: levels.use(t['gradient'], 0.1, 0.8, 1.4, 0.05, 0.95))
    color_matrix = s.ColorMatrix()
    add('color_matrix', color_matrix, lambda t: color_matrix.use(t['gradient'], s.saturation_matrix(0.3)))
    posterize = s.Posterize()
    add('posterize', posterize, lambda t: posterize.use(t['gradient'], 4))
    dither = s.Dither()
    for pattern in s.DitherPattern:
        add(f'dither_{pattern.name.lower()}', dither, lambda t, p=pattern: dither.use(t['gradient'], 3, p))
    vignette = s.Vignette()
    add('vignette', vignette, lambda t: vignette.use(t['gradient'], 0.8, 0.4, 0.5))
    mosaic = s.Mosaic()
    add('mosaic', mosaic, lambda t: mosaic.use(t['gradient'], 16.0))
    displace = s.Displace()
    add('displace', displace, lambda t: displace.use(t['second'], t['gradient'], 0.05))
    median = s.Median()
    add('median', median, lambda t: median.use(t['second']))
    bilateral = s.Bilateral()
    add('bilateral', bilateral, lambda t: bilateral.use(t['second'], 3, 2.0, 0.1))
    chroma_key = s.ChromaKey()
    add('chroma_key', chroma_key, lambda t: chroma_key.use(t['second'], Color(1.0, 0.75, 0.0), 0.1, 0.1))
    luma_key = s.LumaKey()
    add('luma_key', luma_key, lambda t: luma_key.use(t['gradient'], 0.4, 0.2))
    dual = s.DualFilter()
    add('dual_filter_down', dual, lambda t: dual.down(t['second']))
    add('dual_filter_up', dual, lambda t: dual.up(t['second']))

    # The render's shaders the GL plan's line S renames inputs of (references before)
    lanes = TimeSeries()
    colors = [(1.0, 0.45, 0.1, 1.0), (0.1, 0.5, 1.0, 1.0), (0.4, 1.0, 0.5, 1.0)]
    add('time_series', lanes, lambda t: lanes.use(t['gradient'], 3, 64, 0.04, colors))
    window = Window()
    add('feature_window', window, lambda t: window.use(t['gradient'], 128, 4, 0.25, 0.01, 1.0, (0.0, 1.0), colors))
    feature = Feature()
    sample = HarnessFeature()
    add('feature', feature, lambda t: feature.use(sample, colors, 0.02, 0.01, True))
    add_mix = AddMix()
    add('add_mix', add_mix, lambda t: add_mix.use(t['gradient'], t['second'], 0.6))
    composite = LayerComposite()
    add('layer_composite', composite, lambda t: composite.use([t['gradient'], t['second'], t['soft']], [1.0, 0.6, 0.4], 1.0))

    blend = s.Blend()       # O4: every mode of blend_modes.glsl, the gradient over the checker
    for mode in s.BlendShaderMode:
        add(f'blend_{mode.name.lower()}', blend, lambda t, m=mode: blend.use(t['gradient'], t['second'], m, 1.0))

    bloom = Bloom(3)
    def bloom_process(t: dict[str, Texture]) -> None:
        if not bloom.allocated:
            bloom.allocate(SIZE, SIZE)
        bloom.process(t['second'], 0.5)
    add('bloom', None, lambda t: bloom.texture.draw(), bloom_process)
    field = DistanceField()
    def field_process(t: dict[str, Texture]) -> None:
        if not field.allocated:
            field.allocate(SIZE, SIZE)
        field.process(t['hard'])
        exact_distance(field)
    add('distance_field', None, lambda t: field.texture.draw(), field_process)

    add('depth_align', DepthAlign(), None)          # a compute shader: compiled only
    return c


def run(reference: bool) -> int:
    os.makedirs(OUT, exist_ok=True)
    window = context()
    failed: list[str] = []
    try:
        source = inputs()
        target = Fbo()
        target.allocate(SIZE, SIZE, GL_RGBA8)
        table = cases()
        allocated: set[int] = set()
        for name, (shader, draw, prepare) in table.items():
            if shader is not None and id(shader) not in allocated:
                if isinstance(shader, shaders.Lut):
                    shader.allocate()
                    shader.load_cube(LUT)
                else:
                    shader.allocate()
                allocated.add(id(shader))
            if draw is None:
                ok: bool = bool(getattr(shader, 'allocated', False))
                print(f'{name:24s} {"compiled" if ok else "FAILED to compile"}')
                if not ok:
                    failed.append(name)
                continue
            if prepare is not None:
                prepare(source)
            target.begin()
            glViewport(0, 0, SIZE, SIZE)
            Style.push_style()
            Style.set_blend_mode(Style.BlendMode.DISABLED)
            target.clear(0.0, 0.0, 0.0, 0.0)
            draw(source)
            Style.pop_style()
            target.end()
            glFinish()
            image = target.read_to_numpy()
            if image is None:
                failed.append(name)
                print(f'{name:24s} no image')
                continue
            path = os.path.join(OUT, f'{name}.npy')
            cv2.imwrite(os.path.join(OUT, f'{name}.png'), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
            if reference:
                np.save(path, image)
                print(f'{name:24s} reference, mean {image.mean():.1f}')
                continue
            if not os.path.isfile(path):
                np.save(path, image)           # a case new with this step: its first run is its reference
                print(f'{name:24s} new, its reference made, mean {image.mean():.1f}')
                continue
            diff = int(np.abs(image.astype(np.int16) - np.load(path).astype(np.int16)).max())
            print(f'{name:24s} {"ok" if diff <= TOLERANCE else "DIFFERS"}  max {diff}')
            if diff > TOLERANCE:
                failed.append(name)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    print(f'\n{len(failed)} failed' + (f': {", ".join(failed)}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', action='store_true', help='make the reference images')
    sys.exit(run(parser.parse_args().reference))
