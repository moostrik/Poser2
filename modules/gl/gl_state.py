"""gl_state: what GL holds at this moment for a draw or a dispatch, for debugging a pass that comes out wrong.

One call returns the program in use with its uniforms' values, each texture unit's bound texture and size,
the draw framebuffer and its color attachment, the viewport, the blend and the scissor. It
reads GL state, which may wait for the GPU, so it is for debugging, never for a frame's work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from OpenGL.GL import (GL_ACTIVE_TEXTURE, GL_ACTIVE_UNIFORMS, GL_BLEND, GL_BLEND_DST_ALPHA, GL_BLEND_DST_RGB,
                       GL_BLEND_SRC_ALPHA, GL_BLEND_SRC_RGB, GL_COLOR_ATTACHMENT0,
                       GL_CURRENT_PROGRAM, GL_DRAW_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER_BINDING, GL_FLOAT, GL_INT,
                       GL_BOOL, GL_SAMPLER_2D, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, GL_SCISSOR_TEST,
                       GL_TEXTURE0, GL_TEXTURE_2D, GL_TEXTURE_BINDING_2D, GL_TEXTURE_HEIGHT, GL_TEXTURE_WIDTH,
                       GL_VIEWPORT, glActiveTexture, glGetActiveUniform, glGetFramebufferAttachmentParameteriv,
                       glGetIntegerv, glGetProgramiv, glGetTexLevelParameteriv, glGetUniformLocation, glGetUniformfv,
                       glGetUniformiv, glIsEnabled)


def _uniforms(program: int) -> dict[str, Any]:
    """The program's int, bool, sampler and float uniforms with their values."""
    values: dict[str, Any] = {}
    for index in range(int(glGetProgramiv(program, GL_ACTIVE_UNIFORMS))):
        name, _, kind = glGetActiveUniform(program, index)
        name = name.decode() if isinstance(name, bytes) else str(name)
        location = glGetUniformLocation(program, name)
        if location < 0:
            continue
        if kind in (GL_INT, GL_BOOL, GL_SAMPLER_2D):
            value = np.zeros(1, np.int32)
            glGetUniformiv(program, location, value)
            values[name] = int(value[0])
        elif kind == GL_FLOAT:
            value = np.zeros(1, np.float32)
            glGetUniformfv(program, location, value)
            values[name] = float(value[0])
    return values


def gl_state(units: int = 8) -> dict[str, Any]:
    """The GL state a draw or a dispatch would run with now; texture units 0 to `units` − 1."""
    active = int(glGetIntegerv(GL_ACTIVE_TEXTURE))
    bound: list[tuple[int, tuple[int, int] | None]] = []
    for unit in range(units):
        glActiveTexture(GL_TEXTURE0 + unit)
        texture = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
        size = (int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)),
                int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT))) if texture else None
        bound.append((texture, size))
    glActiveTexture(active)
    program = int(glGetIntegerv(GL_CURRENT_PROGRAM))
    framebuffer = int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))
    attachment = int(glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                           GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME)) if framebuffer else 0
    return {
        'program': program,
        'uniforms': _uniforms(program) if program else {},
        'textures': bound,
        'framebuffer': framebuffer,
        'color_attachment': attachment,
        'viewport': [int(v) for v in glGetIntegerv(GL_VIEWPORT)],
        'blend': (bool(glIsEnabled(GL_BLEND)), int(glGetIntegerv(GL_BLEND_SRC_RGB)), int(glGetIntegerv(GL_BLEND_DST_RGB)),
                  int(glGetIntegerv(GL_BLEND_SRC_ALPHA)), int(glGetIntegerv(GL_BLEND_DST_ALPHA))),
        'scissor': bool(glIsEnabled(GL_SCISSOR_TEST)),
    }
