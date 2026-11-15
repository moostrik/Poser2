# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F). Built: the shader check, smile's shaders out of the GL,
one shaders folder, a texture that draws itself and `gl.Context` (described in [GL.md](GL.md)), time from
`perf_counter`, `FPS` over a window in seconds, `Text` asking GL nothing, the shaders' names (G: inputs by role,
camelCase uniforms, `DrawBox`, no dead code). Each step is built on Matthias's "go", which covers the harness runs
the steps name. What the code shows is marked **(seen)**.

The plan changes the GL's form, not what it draws: every app looks as it does now. The harness
(`scripts/shader_check.py`) holds every gl shader, the texture's draws and the text to its reference within 1/255.

Besides their names, `modules/gl`'s shaders follow no shared scheme **(seen)**: a color as four floats or a tuple, a
rectangle as four floats or a `Rect`; device coordinates (−1..1), texture coordinates (0..1) or texture space with an
aspect ratio; the texel size worked out by the shader, given (`texel_size`) or as a `resolution`.

| Step                      | When                                     | Checked by                              |
|---------------------------|------------------------------------------|-----------------------------------------|
| H. Shader coordinates     | on Matthias's "go", after the placement rule | the harness, text's unit test, then Matthias's look |
| I. Capture asks GL nothing | after H                                 | a screen recording                      |
| J. Inputs by type         | after H                                  | the harness                             |
| K. Duplicate shaders      | decided together, once the rest is built | the harness, then Matthias's look       |

## H. Shader coordinates

Two kinds of vertex stage, each in its textbook form:

- A pass fills its whole target and places nothing (the blurs, the blends, every flow step): the quad in GL's own
  −1..1, through `_generic.vert`, which stays as it is. Clip space fits a target of any size without a uniform or the
  context's state, as post-processing chains have it.
- A draw places something in the target: in 0..1 with the origin top-left, as our `Rect`, `texture.draw` and the
  pose points have it, turned into GL's −1..1 by one fixed line from one shared include. No matrix and no size uniform:
  the geometry is worked out on the CPU (poses, rectangles), and 0..1 holds for a target of any size, so a draw or an
  FBO switch sets no state and makes no GL call for its coordinates. The placing stages: `blit.vert`,
  `draw_box.vert`, `draw_colored_rectangle.vert`, `draw_rectangle_outline.vert` and `glyphs.vert` in the vertex
  stage; `DrawCircles` and the render's `PosePointLines` in the fragment stage, whose points sit in `texCoord` with
  an aspect ratio **(seen)**. Today they come in three units: pixels (`draw_box`, `glyphs`), 0..1 top-left (the
  rectangles, the blit, `PosePointLines`) and GL's −1..1 (`DrawCircles`) **(seen)**. Afterwards none flips or
  converts on its own: today `ViewPoseLayer` flips its points before `DrawCircles` because "`PosePointLines` flips
  Y internally" and "`DrawCircles` takes GL coords" **(seen)**.

The flow's volume views (`cube_axes`, `cube_edges`, `glyph_arrows`, `glyph_dots`) and its particles are 3D inside
their own pass and take a `view_projection` of their own **(seen)**; they stay as they are.

`Text` builds its glyph quads on the CPU in pixels and lays them out in the same method that draws them **(seen)**.
The layout goes into a plain module beside it, which turns a string into glyph quads in 0..1 for a target size;
`Text` draws what it returns, and its callers no longer pass a width and height (about 30 `draw_text` and
`draw_box_text` calls). `DrawBox` takes its rectangle in 0..1 the same way. The layout's unit test holds it; the
harness's text case holds the drawing.

The target's size: `Fbo.begin()` and the window's draw note in `gl.Context` the size of what they bind, a value
`Text` reads; nothing is skipped by it. `Fbo.begin()` calls `glViewport` once: today it keeps the viewport in one
global for the whole program, skips a repeat by it, then calls `glViewport` again regardless **(seen)**; the global
goes.

Callers that change with it: `ViewPoseLayer`'s flip, `PosePointLines`' three callers (`pose_line_layer`,
`pose_renderer`, white_space's `pose_figure_layer`), `DrawCircles`' callers, and every text call. The harness
checks the shaders, not these layers; for them it is Matthias's look.

Checked by the harness: before H, `scripts/shader_check.py` gets every shader H changes, the render's and smile's
included, its references made from today's code; after H each matches them within 1/255, then Matthias looks.

## I. Capture asks GL nothing

`ScreenCapture` asks GL its pack alignment at every capture (`glGetIntegerv(GL_PACK_ALIGNMENT)`) to restore it
after its reads **(seen)**. The alignment is set to 1 once for the capture's reads and never asked. `gl_state.py`
and `profiler.py` ask GL by purpose, as debug tools, and stay.

## J. Inputs by type

`modules/gl`'s shaders take a color as four floats or a tuple and a rectangle as four floats or a `Rect`, and work
out the texel size themselves, take it given (`texel_size`) or as a `resolution` **(seen)**. Afterwards a color is a
`Color`, a rectangle a `Rect`, and the texel size is taken from the input texture. The harness holds each shader to
its reference.

## K. Duplicate shaders

What stays and what goes is decided with Matthias once the rest of the plan is built. To look at: `MaskBlur` (one
shader, a direction) against `BlurH` and `BlurV`; `Blend`, `MaskBlend` and `Curtain`, all `(src, dst, fade)`;
`Hsl` against `Hsv`; `Noise` against `NoiseSimplex`; `NoiseBlend` against `NoiseSimplexBlend`; `DrawBox` against
`DrawColoredRectangle`; the flow's own `Blit` (`modules/flow/shaders/blit.py`, `FlowUtil`'s stretch and the bridge's
`CaptureMax`) against the GL's **(seen)**; the window manager's test, which `modules/render/tests/test_window.py` and
`apps/smile/render/tests/test_window.py` hold near line for line **(seen)**.

## Rules it touches

- `CLAUDE.md` API evolution, remove obsolete pathways instead of leaving parallel ones: K
- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: H, I, J, K
- `CLAUDE.md` Performance, prefer the design with the lower latency: H, I
- `CLAUDE.md` API evolution, update all affected call sites in the same change: H, J
- `gl.md` set the GL state a draw needs where it draws: H
- `gl.md` prefer as few GL calls and state switches as possible: H, I
- `gl.md` simple geometry, as from poses, is worked out on the CPU, in plain modules: H
- `gl.md` the placement rule, waiting in the rules thread: H
