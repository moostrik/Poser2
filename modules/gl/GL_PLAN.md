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

| Step                 | When                                 | Checked by                              |
|----------------------|--------------------------------------|-----------------------------------------|
| H. Shader coordinates | a step of its own, planned with Matthias | the harness, then Matthias's look  |
| I. Duplicate shaders | decided together, once the rest is built | the harness, then Matthias's look   |

## H. Shader coordinates

Colors as `Color` and rectangles as `Rect`; texture coordinates 0..1 with the origin top-left wherever a `Rect` is
passed, as the frameworks' 2D drawing, our `Rect` and images have it (the grid: "Rect y is top-down" **(seen)**), and
GL's bottom-left only inside the shaders; the texel size from the input. The origin is turned in one place: a texture
is held in GL's own row order, a CPU image flipped once at its upload (`Image`); one 2D projection the context holds
turns the top-left API into GL's coordinates, as Cinder's `setMatricesWindow(w, h, originUpperLeft = true)` and
openFrameworks' screen setup do, applied by every vertex stage (`_generic.vert`, the rectangles, the circles, the
glyphs, and the render's `PosePointLines`, with its callers). No shader flips on its own: today `ViewPoseLayer` flips
its points before `DrawCircles` because "`PosePointLines` flips Y internally" and "`DrawCircles` takes GL coords"
**(seen)**. Not built until planned on its own with Matthias.

## I. Duplicate shaders

What stays and what goes is decided with Matthias once the rest of the plan is built. To look at: `MaskBlur` (one
shader, a direction) against `BlurH` and `BlurV`; `Blend`, `MaskBlend` and `Curtain`, all `(src, dst, fade)`;
`Hsl` against `Hsv`; `Noise` against `NoiseSimplex`; `NoiseBlend` against `NoiseSimplexBlend`; `DrawBox` against
`DrawColoredRectangle`; the flow's own `Blit` (`modules/flow/shaders/blit.py`, `FlowUtil`'s stretch and the bridge's
`CaptureMax`) against the GL's **(seen)**; the window manager's test, which `modules/render/tests/test_window.py` and
`apps/smile/render/tests/test_window.py` hold near line for line **(seen)**.

## Rules it touches

- `CLAUDE.md` API evolution, remove obsolete pathways instead of leaving parallel ones: I
- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: H, I
