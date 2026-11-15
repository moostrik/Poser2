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
| H. Shader coordinates     | on Matthias's "go"                       | the harness, then Matthias's look       |
| I. Bindings kept          | after H                                  | the harness, the render's CPU time      |
| J. Capture asks GL nothing | after H                                 | a screen recording                      |
| K. Text layout on its own | after H                                  | its unit test, the harness              |
| L. Duplicate shaders      | decided together, once the rest is built | the harness, then Matthias's look       |

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

`Text` builds its glyph quads on the CPU in pixels **(seen)**; it turns them into 0..1 there, from the size of the
bound target it reads on the CPU, so its callers no longer pass a width and height (about 30 `draw_text` and
`draw_box_text` calls). `DrawBox` takes its rectangle in 0..1 the same way.

The bound target is kept in `gl.Context`, as its blend state is: `Fbo.begin()` and the window's draw tell the
context which target and viewport they bind, and the context calls `glViewport` only when the viewport changes.
Today `Fbo.begin()` keeps the viewport in one global for the whole program, skips a repeat by it, then calls
`glViewport` again regardless; the window manager, `Texture2DArray` and smile's renders call `glViewport` past it
**(seen)**, so the global cannot be trusted. They all go through the context, so `Text` reads the target's size there.

Callers that change with it: `ViewPoseLayer`'s flip, `PosePointLines`' three callers (`pose_line_layer`,
`pose_renderer`, white_space's `pose_figure_layer`), `DrawCircles`' callers, and every text call. The harness
checks the shaders, not these layers; for them it is Matthias's look.

Besides the coordinates: colors as `Color`, rectangles as `Rect`, the texel size from the input texture. A texture is
held in GL's own row order, a CPU image flipped once at its upload (`Image`).

Checked by the harness: before H, `scripts/shader_check.py` gets every shader H changes, the render's and smile's
included, its references made from today's code; after H each matches them within 1/255, then Matthias looks.

## I. Bindings kept

Every shader's `use()` calls `glUseProgram`, binds each texture and sets every uniform at each draw, also when
nothing changed since the last **(seen)**. The context keeps the bound program and the texture bound to each unit,
as Cinder's `gl::Context` does, and skips a call that binds what is already bound; a draw still states everything it
needs (gl.md: the state a draw needs is set where it draws, the context skips what is already set). What it saves is
not measured: the render's update and draw CPU time (the window manager's timers) on the same scene before and after
says, and the step stays only if the saving follows from the calls it skips.

## J. Capture asks GL nothing

`ScreenCapture` asks GL its pack alignment at every capture (`glGetIntegerv(GL_PACK_ALIGNMENT)`) to restore it
after its reads **(seen)**. The alignment is set to 1 once for the capture's reads and never asked. `gl_state.py`
and `profiler.py` ask GL by purpose, as debug tools, and stay.

## K. Text layout on its own

`Text.draw_text` lays out its glyphs (positions, atlas coordinates) and draws them in one method **(seen)**, so the
layout cannot be tested without a GL context. The layout goes into a plain module beside it, which turns a string
into glyph quads in 0..1; `Text` draws what it returns. Its unit test holds the layout; the harness's text case
holds the drawing.

## L. Duplicate shaders

What stays and what goes is decided with Matthias once the rest of the plan is built. To look at: `MaskBlur` (one
shader, a direction) against `BlurH` and `BlurV`; `Blend`, `MaskBlend` and `Curtain`, all `(src, dst, fade)`;
`Hsl` against `Hsv`; `Noise` against `NoiseSimplex`; `NoiseBlend` against `NoiseSimplexBlend`; `DrawBox` against
`DrawColoredRectangle`; the flow's own `Blit` (`modules/flow/shaders/blit.py`, `FlowUtil`'s stretch and the bridge's
`CaptureMax`) against the GL's **(seen)**; the window manager's test, which `modules/render/tests/test_window.py` and
`apps/smile/render/tests/test_window.py` hold near line for line **(seen)**.

## Rules it touches

- `CLAUDE.md` API evolution, remove obsolete pathways instead of leaving parallel ones: L
- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: H, I, J, K, L
- `CLAUDE.md` Performance, prefer the design with the lower latency: H, I, J
- `CLAUDE.md` API evolution, update all affected call sites in the same change: H
- `gl.md` set the GL state a draw needs where it draws, through `gl.Context`; it skips what is already set: H, I
- `gl.md` prefer as few GL calls and state switches as possible: H, I, J
- `gl.md` simple geometry, as from poses, is worked out on the CPU, in plain modules: H, K
