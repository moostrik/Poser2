# The GL's plan

The GL held up to the rules (2026-11-15, the apps' plan F). Built: the shader check, smile's shaders out of the GL,
one shaders folder, a texture that draws itself and `gl.Context` (described in [GL.md](GL.md)), time from
`perf_counter`, `FPS` over a window in seconds, `Text` asking GL nothing, the shaders' names (G: inputs by role,
camelCase uniforms, `DrawBox`, no dead code), the shaders' coordinates (H: every 2D position 0..1 from the top left,
one conversion in `position.glsl`), the capture asking GL nothing (I), the inputs by type (J: `Color`, `Rect`, the
texel size from the texture). Each step is built on Matthias's "go", which covers the harness runs the steps name.
What the code shows is marked **(seen)**.

The plan changes the GL's form, not what it draws: every app looks as it does now. The harness
(`scripts/shader_check.py`) holds every gl shader, the shaders outside the GL that take a position, the texture's
draws and the text to its reference within 1/255.

| Step                      | When                                     | Checked by                              |
|---------------------------|------------------------------------------|-----------------------------------------|
| K. Duplicate shaders      | decided together, now the rest is built  | the harness, then Matthias's look       |

## K. Duplicate shaders

What stays and what goes is decided with Matthias now the rest of the plan is built. To look at: `MaskBlur` (one
shader, a direction) against `BlurH` and `BlurV`; `Blend`, `MaskBlend` and `Curtain`, all `(src, dst, fade)`;
`Hsl` against `Hsv`; `Noise` against `NoiseSimplex`; `NoiseBlend` against `NoiseSimplexBlend`; `DrawBox` against
`DrawColoredRectangle`, both now `(Rect, Color)` **(seen)**; `DrawRoi` (without a turn) against
`texture.draw(src=…)` **(inferred)**; the flow's own `Blit` (`modules/flow/shaders/blit.py`, `FlowUtil`'s stretch and
the bridge's `CaptureMax`) against the GL's **(seen)**; the window manager's test, which
`modules/render/tests/test_window.py` and `apps/smile/render/tests/test_window.py` hold near line for line **(seen)**.

## Rules it touches

- `CLAUDE.md` API evolution, remove obsolete pathways instead of leaving parallel ones: K
- `CLAUDE.md` Running code, run a GPU harness only on the user's go, and only what the plan names: K
