# GL plan

Proposed by the package scan of 2026-11-21, on the code at c5d86a7d, with the rules thread's sanity check and the
plan analysis folded in. The why of each step is the finding it rests on, marked seen (in the code) or deduction.
Every step is one commit and is checked by pyright, the guard tests and gl's tests, and by the runs it names;
shader_check compares with references made at HEAD before the first step. A run happens at its step, after a
tasklist check that no app is running, unless Matthias says he is using the PC or the app.


## C. Text's font in its package

- C1. `RobotoMono-Regular.ttf` moves from `data/` into `gl/text` with its licence (Apache 2.0) beside it, and `Text`
  finds it beside `text.py`; the three pose layers that pass its path (`render/modules/pose/feature_frame_layer.py`,
  `feature_window_layer.py`, `motion_time_renderer.py`) take the default. Why: `Text.DEFAULT_FONT` is
  `data/RobotoMono-Regular.ttf`, found from the working directory, outside the package that uses it, and no licence
  for it is in the repo (seen). Check: shader_check (Text); a text draw in hd_trio -sim (the pose layers' text).

## D. Every GL object from gl's classes

- D1. `PixelBufferPool` in `texture/pixel_buffer_pool.py`, exported from `gl`: a pool of pixel buffers with their
  fences, for a readback (GL_PIXEL_PACK_BUFFER) or an upload (GL_PIXEL_UNPACK_BUFFER); it makes, maps (persistently,
  or not at all) and frees the buffers, and holds their fences: which slot is free, which is ready. The transfer
  stays with each caller: `glReadPixels`, the upload, the copy out. In the same step gl moves its three users onto
  it: `flow/bridge/velocity_peak.py` (`_buffers`, `_copy`, the read of the bins),
  `render/general/screen_recording/screen_capture.py` (its pool) and `screen_preview.py` (its upload buffers), and asks
  the render thread for feedback afterwards (RENDER_PLAN.md points here). Why: gl.md "A texture, framebuffer or
  buffer is made by one of gl's classes"; these three make their buffers with `glGenBuffers`, give them storage
  while bound and keep their own fences, each its own pool (seen); a class that also did the three transfers would
  need an option per caller (deduction). Check: a -sim with the screen recorder on (capture and preview), and
  deep_flow -sim (the bridge's velocity peak).
- D2. `MultiTargetFbo` in `texture/multi_target_fbo.py`, exported from `gl`: a framebuffer that draws into the
  textures it is given, two or more at once (`begin`/`end`, its draw buffers set at `allocate`), and owns no
  texture; `Fbo`, one texture that draws into itself, stays as it is. `flow/fluid_3d/particle_density.py`
  (`_weighted_fbo`, `_surface_fbo`, two targets each) uses it, and `flow/motion/depth_velocity.py` one per side of its
  ping-pong (`_estimate` with `misses[0]`, `_estimate.back` with `misses[1]`), swapped together, in place of the
  `misses` it attaches to each buffer of a `SwapFbo` by `fbo_id`. Why: the same gl.md line; particle_density makes
  two framebuffers with `glGenFramebuffers`, and depth_velocity attaches a second target to a `gl.Fbo` from outside
  gl, with its own `glDrawBuffers` (seen); extra targets on `Fbo` would mix its own texture with other ones in
  `begin`, `clear` and `deallocate` (deduction). Check: deep_flow -sim (the particles, the motion).

GL.md says what each step adds where it is described: `upload` and `bgr` under the textures, `PixelBufferPool` and
`MultiTargetFbo` beside them, the font's place under text. BRIDGE_3D.md is read for `velocity_peak` with D1.

Form, done only in a commit that opens the file anyway, never as a step of its own: PEP 8 in `fbo.py` and
the imports of `image.py` and `tensor.py` (with A1 to A4); the docstrings of `Texture3D` and `Texture2DArray` (with
A1), `ComputeShader` (with B1) and `Text` (with C1) say what the thing is, the 3D-tiling reason going to GL.md; the
two swap tests become one `test_swap.py` in a commit that changes `Swap` or them.

## Questions

- `gl.shaders.WeightedSum` and `flow.bridge_3d`'s `WeightedSum` are two operations under one name (the flow's is the
  bridge's mix of up to seven sources): the flow's renamed after what it does? The flow thread's call.

## Rules this plan touches

- CLAUDE.md: Inside a package, code that does the same is written once
- CLAUDE.md: Similar things share the noun and differ by a qualifier (`Frame`, `CameraFrame`)
- CLAUDE.md: Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: Update all affected call sites in the same change
- CLAUDE.md: A docstring says what the thing is
- CLAUDE.md: Reasons sit in the package's document, history in the commit
- CLAUDE.md: Use Python 3.12 and follow PEP 8
- CLAUDE.md: A plan step that is checked by a run or a test is one commit
- CLAUDE.md: Run a test of the app or a GPU harness only on the user's go, and only what the plan names
- gl.md: A texture, framebuffer or buffer is made by one of gl's classes
- gl.md: Pair every creation with its destruction
- gl.md: Shaders get checked in a GPU harness
- docs.md: A component is described in its own document; other documents link to it
- plans.md: A change that is already worked out is costly to get: keep it
