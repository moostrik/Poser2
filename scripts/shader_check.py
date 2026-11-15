"""The shader check: every shader of `modules/gl` drawn once on known input and held to a reference image.

A hidden GL 4.6 window gives the context; it is this script's own render thread. Each case draws into a 128 x 128
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

import cv2
import glfw
import numpy as np
from OpenGL.GL import (GL_R8, GL_RED, GL_RGBA, GL_RGBA8, GL_TEXTURE_2D, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE,
                       glBindTexture, glFinish, glPixelStorei, glTexSubImage2D, glViewport)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apps.smile.render.shaders import FlashIn, FlashOut, Omission  # noqa: E402
from modules.gl import Context, Fbo, Style, Text, Texture, shaders  # noqa: E402
from modules.gl.shaders.draw_box import DrawBox  # noqa: E402
from modules.pose import PointLandmark, Points2D  # noqa: E402
from modules.render import PosePointLines  # noqa: E402
from modules.utils import Point2f, Rect  # noqa: E402

OUT: str = os.path.join('temp', 'shader_check')
SIZE: int = 128
TOLERANCE: int = 1                  # per channel, of 255
LUT: str = os.path.join('data', 'luts', 'Blockbuster.cube')


def context() -> object:
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


def inputs() -> dict[str, Texture]:
    """The known inputs: a color gradient, a second image, a soft round mask, and a hard-edged one."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / (SIZE - 1)
    gradient = np.dstack([x, y, 1.0 - x, 0.5 + 0.5 * x]) * 255
    checker = ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(np.float32)
    second = np.dstack([checker, 0.5 * checker + 0.25, 1.0 - checker, np.ones_like(x)]) * 255
    radius = np.hypot(x - 0.5, y - 0.5)
    soft = np.clip(1.0 - radius * 2.5, 0.0, 1.0) * 255
    hard = (radius < 0.3).astype(np.float32) * 255
    return {'gradient': texture(gradient.astype(np.uint8)), 'second': texture(second.astype(np.uint8)),
            'soft': texture(soft.astype(np.uint8)), 'hard': texture(hard.astype(np.uint8))}


def pose() -> Points2D:
    """A standing figure, every point found."""
    values = np.full((len(PointLandmark), 2), 0.5, dtype=np.float32)
    for i, landmark in enumerate(PointLandmark):
        values[i] = (0.3 + 0.4 * ((i * 7) % 5) / 4, 0.1 + 0.8 * i / (len(PointLandmark) - 1))
    return Points2D(values, np.ones(len(PointLandmark), dtype=np.float32))


Case = Callable[[dict[str, Texture]], None]


def cases() -> dict[str, tuple[object, Case]]:
    """Each case: the shader it allocates (None for one that is not a shader) and its draw."""
    s = shaders
    c: dict[str, tuple[object, Case]] = {}

    def add(name: str, shader: object, draw: Case) -> None:
        c[name] = (shader, draw)

    # The texture's draws, through the context's Blit (no shader of their own to allocate)
    add('blit', None, lambda t: t['gradient'].draw())
    add('blit_opacity', None, lambda t: t['gradient'].draw(opacity=0.5))
    add('blit_flip_y', None, lambda t: t['gradient'].draw(src=Rect(0.0, 1.0, 1.0, -1.0)))
    add('blit_flip_x', None, lambda t: t['gradient'].draw(src=Rect(1.0, 0.0, -1.0, 1.0)))
    add('blit_region', None, lambda t: t['gradient'].draw(src=Rect(0.25, 0.25, 0.5, 0.5)))
    add('draw_dst', None, lambda t: t['gradient'].draw(dst=Rect(0.1, 0.2, 0.5, 0.4)))
    add('draw_dst_src', None, lambda t: t['second'].draw(Rect(0.5, 0.0, 0.5, 1.0), Rect(0.0, 0.5, 0.5, 0.5)))

    contrast = s.Contrast()
    add('contrast', contrast, lambda t: contrast.use(t['gradient'], 0.1, 1.5))
    exposure = s.Exposure()
    add('exposure', exposure, lambda t: exposure.use(t['gradient'], 1.5, 0.05, 0.8))
    hsl = s.Hsl()
    add('hsl', hsl, lambda t: hsl.use(t['gradient'], 0.2, 1.3, 0.1))
    hsv = s.Hsv()
    add('hsv', hsv, lambda t: hsv.use(t['gradient'], 0.2, 1.3, 1.1))
    lut = s.Lut()
    add('lut', lut, lambda t: lut.use(t['gradient'], 0.8))
    noise = s.Noise()

    def draw_noise(_: dict[str, Texture]) -> None:
        random.seed(0)
        noise.use()
    add('noise', noise, draw_noise)
    simplex = s.NoiseSimplex()

    add('noise_simplex', simplex, lambda t: simplex.use(12.5, 1.0, 0.5))
    nv12 = s.Nv12()
    add('nv12_y', nv12, lambda t: nv12.use(t['gradient'], s.Nv12Plane.Y))
    add('nv12_uv', nv12, lambda t: nv12.use(t['gradient'], s.Nv12Plane.UV))
    sharpen = s.Sharpen()
    add('sharpen', sharpen, lambda t: sharpen.use(t['second'], 1.5))

    blur_h, blur_v = s.BlurH(), s.BlurV()
    add('blur_h', blur_h, lambda t: blur_h.use(t['second'], 3.0))
    add('blur_v', blur_v, lambda t: blur_v.use(t['second'], 3.0))
    curtain = s.Curtain()
    add('curtain', curtain, lambda t: curtain.use(t['gradient'], t['second'], 0.4))
    flash_in = FlashIn()
    add('flash_in', flash_in, lambda t: flash_in.use(t['gradient'], t['second'], 0.7))
    flash_out = FlashOut()
    add('flash_out', flash_out, lambda t: flash_out.use(t['gradient'], t['second'], t['soft'], 0.6))
    noise_blend = s.NoiseBlend()
    add('noise_blend', noise_blend, lambda t: noise_blend.use(t['gradient'], t['second'], t['soft'], 0.5))
    simplex_blend = s.NoiseSimplexBlend()
    add('noise_simplex_blend', simplex_blend,
        lambda t: simplex_blend.use(t['gradient'], t['second'], t['soft'], 0.5))
    omission = Omission()
    add('omission', omission, lambda t: omission.use(t['gradient'], t['second'], t['soft'], 0.3, 0.6))

    blend = s.Blend()
    add('blend', blend, lambda t: blend.use(t['gradient'], t['second'], 0.3))
    tint = s.Tint()
    add('tint', tint, lambda t: tint.use(t['soft'], 0.9, 0.4, 0.1, 1.0))
    draw_roi = s.DrawRoi()
    add('draw_roi', draw_roi, lambda t: draw_roi.use(t['gradient'], Rect(0.2, 0.1, 0.5, 0.6)))
    add('draw_roi_turned', draw_roi,
        lambda t: draw_roi.use(t['gradient'], Rect(0.2, 0.1, 0.5, 0.6), 0.4, Point2f(0.45, 0.4), 1.0))
    circles = s.DrawCircles()
    add('draw_circles', circles,
        lambda t: circles.use([(0.25, 0.25), (0.6, 0.7), (0.8, 0.3)], 0.08, 0.01, (1.0, 0.5, 0.2, 1.0), 1.0))
    rect = s.DrawColoredRectangle()
    add('draw_colored_rectangle', rect, lambda t: rect.use(0.1, 0.2, 0.5, 0.3, 0.2, 0.8, 0.4, 0.9))
    outline = s.DrawRectangleOutline()
    add('draw_rectangle_outline', outline, lambda t: outline.use(0.1, 0.2, 0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 0.02, 0.03))

    mask_apply = s.MaskApply()
    add('mask_apply', mask_apply, lambda t: mask_apply.use(t['gradient'], t['soft'], 1.0))
    mask_blend = s.MaskBlend()
    add('mask_blend', mask_blend, lambda t: mask_blend.use(t['soft'], t['hard'], 0.4))
    mask_blur = s.MaskBlur()
    add('mask_blur_h', mask_blur, lambda t: mask_blur.use(t['hard'], True, 3.0))
    add('mask_blur_v', mask_blur, lambda t: mask_blur.use(t['hard'], False, 3.0))
    dilate, erode = s.MaskDilate(), s.MaskErode()
    add('mask_dilate', dilate, lambda t: dilate.use(t['hard'], 3.0))
    add('mask_erode', erode, lambda t: erode.use(t['hard'], 3.0))
    hard_step = s.MaskHardStep()
    add('mask_hard_step', hard_step, lambda t: hard_step.use(t['soft'], 0.4))
    edge = s.EdgeSketch()
    add('edge_sketch', edge, lambda t: edge.use(t['second'], 0.1, 1.0, False))
    detail = s.DetailThreshold()
    add('detail_threshold', detail, lambda t: detail.use(t['second'], 0.5, 1.0, 3.0, False))

    box = DrawBox()
    add('box', box, lambda t: box.use((10.0, 20.0, 60.0, 30.0), (0.2, 0.4, 0.9, 0.8), (float(SIZE), float(SIZE))))
    lines = PosePointLines()
    figure = pose()
    add('pose_point_lines', lines, lambda t: lines.use(figure, 0.02, 0.005, (1.0, 0.8, 0.2, 1.0), True))

    text = Text()

    def draw_text(_: dict[str, Texture]) -> None:
        text.draw_text(4, 8, 'Poser 0123', (1.0, 1.0, 1.0, 1.0), SIZE, SIZE)
        text.draw_box_text(4, 60, 'box 45', (1.0, 0.9, 0.2, 1.0), (0.0, 0.0, 0.3, 0.7), SIZE, SIZE)
    add('text', text, draw_text)

    add('depth_align', s.DepthAlign(), None)        # a compute shader: compiled only
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
        for name, (shader, draw) in table.items():
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
