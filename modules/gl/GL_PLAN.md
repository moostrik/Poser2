# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F). Built: the shader check (0), smile's shaders out of the GL
(A), one shaders folder (B), a texture that draws itself and `gl.Context` (C, described in [GL.md](GL.md)), time from
`perf_counter` (D), `FPS` over a window in seconds (D2), `Text` asking GL nothing (F), the names (E). Each step is built on Matthias's "go", which covers the harness runs the steps name.
What the code shows is marked **(seen)**.

The plan changes the GL's form, not what it draws: every app looks as it does now. The harness
(`scripts/shader_check.py`) holds every gl shader, the texture's draws and the text to its reference within 1/255.

| Commit                  | Checked by                                  |
|-------------------------|---------------------------------------------|
| G1                      | the harness                                 |
| G2, its own step        | the harness, then Matthias's look           |
| G3, decided together    | the harness, then Matthias's look           |

## G. The shaders on one scheme

For `modules/gl`'s shaders. They follow no shared scheme **(seen)**: textures named by slot (`tex0` 19 times,
`tex1`, `tex2`), `tex` (8) or role (`src`, `dst`, `mask`); uniforms mostly camelCase, `box_color`, `text_color`,
`num_circles` and `aspect_ratio` snake_case; a color as four floats or a tuple, a rectangle as four floats or a
`Rect`; device coordinates (−1..1), texture coordinates (0..1) or texture space with an aspect ratio; the texel size
worked out by the shader, given (`texel_size`) or as a `resolution`; commented-out code in 20 `.frag` files.

**G1. Names and no dead code.** Each shader's inputs are named by role (`src`, `dst`, `mask`), in the `.frag` and in
`use()` alike; its uniforms are camelCase; commented-out code goes. `box_shader` loses its `Shader` ending. Built
now, after F; then the plan stops.

**G2. The coordinate system, a step of its own.** Colors as `Color` and rectangles as `Rect`; texture coordinates
0..1 with the origin top-left wherever a `Rect` is passed, as the frameworks' 2D drawing, our `Rect` and images have
it (the grid: "Rect y is top-down" **(seen)**), and GL's bottom-left only inside the shaders; the texel size from the
input. The origin is turned in one place: a texture is held in GL's own row order, a CPU image flipped once at its
upload (`Image`); one 2D projection the context holds turns the top-left API into GL's coordinates, as Cinder's
`setMatricesWindow(w, h, originUpperLeft = true)` and openFrameworks' screen setup do, applied by every vertex stage
(`_generic.vert`, the rectangles, the circles, the glyphs, and the render's `PosePointLines`, with its callers). No
shader flips on its own: today `ViewPoseLayer` flips its points before `DrawCircles` because "`PosePointLines` flips Y
internally" and "`DrawCircles` takes GL coords" **(seen)**. Not built until planned on its own with Matthias.

**G3. None twice, decided together.** What stays and what goes is decided with Matthias once the rest of the plan is
built. To look at: `MaskBlur` (one shader, a direction) against `BlurH` and `BlurV`; `Blend`, `MaskBlend` and
`Curtain`, all `(tex0, tex1, fade)`; `Hsl` against `Hsv`; `Noise` against `NoiseSimplex`; `NoiseBlend` against
`NoiseSimplexBlend`; `BoxShader` against `DrawColoredRectangle`; the flow's own `Blit` (`modules/flow/shaders/blit.py`,
`FlowUtil`'s stretch and the bridge's `CaptureMax`) against the GL's **(seen)**.

## Rules it touches

- `CLAUDE.md` Naming, one term per concept: G1
- `CLAUDE.md` API evolution, remove obsolete pathways instead of leaving parallel ones: G3
- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: G
- `gl.md`, a shader is named after the operation it does: G1
