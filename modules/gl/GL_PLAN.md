# GL plan

What is not built in `modules/gl`. A step is one commit, built on Matthias's go, checked by pyright, the guard
tests, gl's tests and the runs it names; a run happens at its step, after a tasklist check that no app is running,
unless Matthias says he is using the PC or the app.

## H. One framebuffer class

- H1. `Fbo` and `MultiTargetFbo` become one `Fbo` with one or more color targets it owns, as Cinder's `gl::Fbo` and
  openFrameworks' `ofFbo`: `allocate(width, height, format)` is the one target, a sequence of formats gives one target
  each; `targets[i]` is target i for every i, `targets[0]` the `Fbo` itself; the optional depth buffer stays. Unlike
  theirs, the `Fbo` is its target 0, a texture: in a 2D texture pipeline a pass writes a texture, and a swap pair is
  used as its current texture, never through one of its buffers (gl.md). `SwapFbo` and `SwapMultiTargetFbo` become one
  swap pair of it, with F1's `target(i)`. Every caller moves with it: `Image` (an `Fbo` with its own `allocate`), the
  ones with several targets (depth_velocity, particle_density), and the ones with one only if they name its format by
  keyword (93 files name `Fbo` or `SwapFbo`: 35 in the apps, 53 in the modules, 5 in the scripts). `LayeredFbo` stays
  as it is: one target of another texture kind (a `Texture2DArray`), attached with all its layers and drawn per layer
  by `gl_Layer`, for the 3D steps; making the 2D `Fbo` generic over the texture kind would weigh on every 2D pass
  (deduction). Latency: none added or taken, the targets and draw buffers are set at allocation only (seen). Why: two
  classes for one thing, a framebuffer that owns its color targets, differing only in their count; the textbook class
  is one (seen: `fbo.py`, `multi_target_fbo.py`; Cinder's `Fbo.h`, openFrameworks' `ofFbo.h`). Check: pyright, the
  guard and gl tests, shader_check, and a -sim of each app (deep_flow with the particles on the weighted look and on
  SURFACE, low_flow, hd_trio, white_space, smile), no GL error.

## Rules this plan touches

- CLAUDE.md: Inside a package, code that does the same is written once
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: Update all affected call sites in the same change
- CLAUDE.md: A plan step that is checked by a run or a test is one commit
- gl.md: Judge a GL design by the latency it adds to the 2D path, camera to screen
- gl.md: A texture, framebuffer or buffer is made by one of gl's classes
- gl.md: Keep a swap pair itself, never one of its buffers; read `current` and `back` where they are used
- gl.md: Shaders get checked in a GPU harness
- docs.md: When renaming or removing a section, state or setting, search the documents for references to it
