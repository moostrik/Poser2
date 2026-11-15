# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F). Built: the shader check, smile's shaders out of the GL,
one shaders folder, a texture that draws itself and `gl.Context` (described in [GL.md](GL.md)), time from
`perf_counter`, `FPS` over a window in seconds, `Text` asking GL nothing, the shaders' names (G: inputs by role,
camelCase uniforms, `DrawBox`, no dead code), the shaders' coordinates (H: every 2D position 0..1 from the top left,
one conversion in `position.glsl`), the capture asking GL nothing (I). Each step is built on Matthias's "go", which covers the harness runs the steps
name. What the code shows is marked **(seen)**.

The plan changes the GL's form, not what it draws: every app looks as it does now. The harness
(`scripts/shader_check.py`) holds every gl shader, the shaders outside the GL that take a position, the texture's
draws and the text to its reference within 1/255.

| Step                      | When                                     | Checked by                              |
|---------------------------|------------------------------------------|-----------------------------------------|
| J. Inputs by type         | on the "go" of 2026-11-15                | the harness, each app with `-sim`       |
| K. Duplicate shaders      | decided together, once the rest is built | the harness, then Matthias's look       |

## J. Inputs by type

`modules/gl`'s shaders take a color as four floats or a tuple and a rectangle as four floats or a `Rect`, and work
out the texel size themselves, take it given (`texel_size`) or as a `resolution` **(seen)**. Afterwards a color is a
`Color`, a rectangle a `Rect`, and the texel size is taken from the input texture; no caller passes `texel_size`
today **(seen)**. The callers that change **(seen)**: `DrawColoredRectangle`'s and `DrawRectangleOutline`'s, which
pass `x, y, w, h, r, g, b, a` (11 calls in 9 files), `Tint`'s one, `DrawCircles`' colors from `ViewPoseLayer`, and
`DrawBox` and `Glyphs` through `Text`, whose 25 callers pass a `Color` for the text and the box.
The harness holds each shader to its reference, its cases rewritten to the new types; each app runs with `-sim`
before and after, its screens compared.

## K. Duplicate shaders

What stays and what goes is decided with Matthias once the rest of the plan is built. To look at: `MaskBlur` (one
shader, a direction) against `BlurH` and `BlurV`; `Blend`, `MaskBlend` and `Curtain`, all `(src, dst, fade)`;
`Hsl` against `Hsv`; `Noise` against `NoiseSimplex`; `NoiseBlend` against `NoiseSimplexBlend`; `DrawBox` against
`DrawColoredRectangle`; `DrawRoi` (without a turn) against `texture.draw(src=…)` **(inferred)**; the flow's own `Blit`
(`modules/flow/shaders/blit.py`, `FlowUtil`'s stretch and the bridge's `CaptureMax`) against the GL's **(seen)**; the
window manager's test, which `modules/render/tests/test_window.py` and `apps/smile/render/tests/test_window.py` hold
near line for line **(seen)**.

## Rules it touches

- `CLAUDE.md` API evolution, remove obsolete pathways instead of leaving parallel ones: K
- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: J, K
- `CLAUDE.md` API evolution, update all affected call sites in the same change: J
