# The GL's plan

Found while the render's rules were ordered; not started. Each step is built on Matthias's "go".

## 1. The order of `modules/gl/shaders`

The render's shaders without a layer come in as they are, with the render's code move. Then to decide:

- What `convenience/` is for
- Whether the shaders are grouped by kind of work (mask, color, draw, blit)

## 2. The shaders' names

The render's rule is that a shader is named after the operation it does and carries no `Shader` ending. `box_shader` and `text_shader` carry it.

## 3. How the GL hands out its shaders

By bare name today (`from modules.gl import Blit`). To decide: whether that still reads well with many more shaders.

## Rules it touches

- `gl.md`: 1, 2, 3
- `CLAUDE.md` Naming: 2
- `CLAUDE.md` Imports: 3
