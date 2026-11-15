# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F). Built: the shader check, smile's shaders out of the GL,
one shaders folder, a texture that draws itself and `gl.Context` (described in [GL.md](GL.md)), time from
`perf_counter`, `FPS` over a window in seconds, `Text` asking GL nothing, the shaders' names (G: inputs by role,
camelCase uniforms, `DrawBox`, no dead code). Each step is built on Matthias's "go", which covers the harness runs
the steps name. What the code shows is marked **(seen)**.

The plan changes the GL's form, not what it draws: every app looks as it does now. The harness
(`scripts/shader_check.py`) holds every gl shader, the texture's draws and the text to its reference within 1/255.

| Step                      | When                                     | Checked by                              |
|---------------------------|------------------------------------------|-----------------------------------------|
| H. Shader coordinates     | on Matthias's "go"                       | the harness, text's unit test, each app before and after |
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
  FBO switch sets no state and makes no GL call for its coordinates. 0..1 counts in the viewport: smile's wall and
  credits place by setting `glViewport` to a part of the target **(seen)**.

Every shader that takes a position **(seen: their uniforms)**:

| Package | Shader                                                    | Takes                        | Today                        |
|---------|-----------------------------------------------------------|------------------------------|------------------------------|
| gl      | `Blit`, `DrawColoredRectangle`, `DrawRectangleOutline`    | a rectangle, vertex stage    | 0..1 top-left                |
| gl      | `DrawBox`, `Glyphs`                                       | a rectangle, vertex stage    | pixels                       |
| gl      | `DrawCircles`                                             | points, fragment stage       | GL's −1..1                   |
| gl      | `DrawRoi`                                                 | a region, fragment stage     | texture coordinates          |
| render  | `PosePointLines`                                          | points and a rectangle       | 0..1 top-left, flipped inside |
| smile   | `FaceColor`                                               | the face's box               | 0..1 top-left                |
| flow    | `SetRegion`, `AddRegion`, `ChannelSetRegion`, `ChannelAddRegion` | a region             | "UV coordinates"             |
| flow    | `DepthUpscale`                                            | the view's window (`roi`)    | texture coordinates          |
| flow    | `ColormapKey`                                             | a box                        | pixels from the bottom left  |

Afterwards each takes 0..1 top-left and none flips or converts on its own: today `ViewPoseLayer` flips its points
before `DrawCircles` because "`PosePointLines` flips Y internally" and "`DrawCircles` takes GL coords" **(seen)**.
Where "texture coordinates" or "UV" means bottom-left is read from each shader before it changes.

The flow's volume views (`cube_axes`, `cube_edges`, `glyph_arrows`, `glyph_dots`) and its particles are 3D inside
their own pass and take a `view_projection` of their own **(seen)**; they stay as they are.

`Text` builds its glyph quads on the CPU in pixels and lays them out in the same method that draws them **(seen)**.
The layout goes into a plain module beside it, which turns a string into glyph quads in 0..1 for the size its caller
gives, as today; `Text` draws what it returns. `DrawBox` takes its rectangle in 0..1 the same way. The layout's unit
test holds it; the harness's text case holds the drawing.

`Fbo.begin()` calls `glViewport` once: today it keeps the viewport in one global for the whole program, skips a
repeat by it, then calls `glViewport` again regardless **(seen)**; the global goes.

Callers that change with it **(seen)**: `ViewPoseLayer`'s flip, the only caller of `DrawCircles`; `PosePointLines`'
three callers (`pose_line_layer`, `pose_renderer`, white_space's `pose_figure_layer`); the callers of `DrawRoi`,
`FaceColor`, the four region shaders, `DepthUpscale` and `ColormapKey`. The rectangles' and the blit's callers already
pass 0..1 top-left and stay; so do the 25 text calls.

Checked by:
- the harness: it already holds the gl shaders and `PosePointLines` to references from before the plan **(seen)**;
  before H it gets `FaceColor` and the flow's six, references from today's code; after H each matches within 1/255
- the text layout's unit test
- each app run before and after H, its screens compared by me; Matthias gets only what I cannot judge

## I. Capture asks GL nothing

`ScreenCapture` asks GL its pack alignment at every capture (`glGetIntegerv(GL_PACK_ALIGNMENT)`) to restore it
after its reads **(seen)**. The alignment is set to 1 once for the capture's reads and never asked. `gl_state.py`
and `profiler.py` ask GL by purpose, as debug tools, and stay.

## J. Inputs by type

`modules/gl`'s shaders take a color as four floats or a tuple and a rectangle as four floats or a `Rect`, and work
out the texel size themselves, take it given (`texel_size`) or as a `resolution` **(seen)**. Afterwards a color is a
`Color`, a rectangle a `Rect`, and the texel size is taken from the input texture; no caller passes `texel_size`
today **(seen)**. The harness holds each shader to its reference.

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
- `gl.md` set the GL state a draw needs where it draws (blend, depth, viewport): H
- `gl.md` prefer as few GL calls and state switches as possible: H, I
- `gl.md` simple geometry, as from poses, is worked out on the CPU, in plain modules: H
- `gl.md` all shaders take positions from 0 to 1, with (0, 0) at the top left: H
- `gl.md` the conversion to OpenGL's own coordinates is written once, and every shader uses it: H
