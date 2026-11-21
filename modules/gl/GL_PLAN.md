# GL plan

What is not built in `modules/gl`. A step is one commit, built on Matthias's go, checked by pyright, the guard
tests, gl's tests and the runs it names; a run happens at its step, after a tasklist check that no app is running,
unless Matthias says he is using the PC or the app.

## E. Every pixel transfer sets its own alignment

- E1. `Tensor._update_with_pbo` and `Tensor._update_with_cpu` set their own unpack alignment at the transfer, as
  `Texture.upload` does; `TextureBase._read` sets its own pack alignment, and `Texture.read_to_numpy` takes a mip
  level (`level`, 0 by default; the 3D and array reads stay at level 0, as an array keeps its layers at every
  level). `flow/motion/solve_capture.read_level` reads through it with `flip=False` (it keeps GL's row order), in
  place of its own bind and `glGetTexImage`; its `kind` and `dtype` go, as its R8 and depth reads are the textures'
  own types. `apps/deep_flow/scripts/view_check.read` reads through it where its channels are the texture's; should
  the level turn out more than a small change, they set the pack alignment at their own read instead.
  `flow/bridge/velocity_peak` stays as it is (one row of 256 × R32UI, the same bytes at any alignment). In the same
  commit, `render/general/screen_recording/screen_capture.py` and `screen_preview.py` lose `_PACK_ALIGNMENT`,
  `_UNPACK_ALIGNMENT` and their set-back to 4 after their own transfers, which kept GL's default for the transfers
  that did not set theirs; the render thread checks those two files afterwards. Why: these transfers rely on the
  alignment GL has at the moment, 4 by default or 1 as `Texture.upload` and `Image` leave it, so a row whose bytes
  are not a multiple of 4 (an R8 or RGB8 texture of odd width) comes out sheared at 4 (seen: the transfers, a scan of
  2026-11-21; the shear, deduction); a read-back is written once (Motion's review of 46df3c07). Check: shader_check
  (`read_to_numpy`); deep_flow -sim (Tensor's uploads); a -sim with the screen recorder making a take and playing it
  back (capture and preview). `solve_capture` (only with `POSER_SOLVE_CAPTURE` set) and `view_check` (a script) are
  not run.

## F. A framebuffer of several targets owns them, and swaps as one

- F1. On c40e354c (the Fluid3D thread's surface color, a third target cleared each frame). `MultiTargetFbo` owns
  its targets, as `Fbo` owns its one texture: `allocate(width, height, formats)` makes a texture per format and hands
  each out by index (`targets[i]`, a `Texture`). In `texture/swap.py`, `Swap[T]` becomes the pair itself
  (`current`, `back`, `swap`, `deallocate`, over anything allocated and freed) and `TextureSwap[T]` the texture
  pair on top of it, with today's forwarding (`width`, `tex_id`, `bind`, `clear`, `set_wrap`, …), the base of
  `SwapFbo`, `SwapTexture3D` and `SwapTexture2DArray`. In `texture/multi_target_fbo.py`, beside `MultiTargetFbo`,
  `SwapMultiTargetFbo` is a `Swap` of two with its own `allocate(width, height, formats)`, and `target(i)` hands out
  target i as a `SwapTarget`: a `Texture` that is always the current framebuffer's target i, as a `SwapFbo` is always
  its current buffer, so a port keeps it. `gl` exports `MultiTargetFbo`, `SwapMultiTargetFbo` and `SwapTarget`.
  `flow/motion/depth_velocity.py` holds its estimate and its misses as one `SwapMultiTargetFbo` (RGBA32F, R16F): it
  draws into `current`, reads `back`, and its `estimate` port and `misses` are `target(0)` and `target(1)`, so
  `_misses_of`, `_targets_of` and the lookup by `fbo_id` go. `flow/fluid_3d/particle_density.py`'s weighted look and
  surface each own their targets (RGBA32F and RGBA16F; R32F, R32F and RGBA16F, as c40e354c allocates them), the
  next passes reading them by index. GL.md's `MultiTargetFbo` paragraph says it owns its targets and swaps as one.
  Why: since 46df3c07 each of depth_velocity's two `MultiTargetFbo`s keeps one raw buffer of the estimate's swap pair
  (`_targets` from `(_estimate.current, _estimate.back)`), picked by the estimate's id each frame, against gl.md's
  "Keep a swap pair itself, never one of its buffers"; a port is "the object itself, never a buffer taken from it"
  (`flow/port.py`); a framebuffer owning its color targets is the textbook shape (Cinder's and openFrameworks' Fbo)
  (seen: `multi_target_fbo.py`, `depth_velocity.py`; only particle_density reads its targets, no port holds them;
  nothing else keeps a buffer of a swap pair, a search of 2026-11-21). Check: pyright, the guard, gl and flow tests;
  deep_flow -sim with the motion, and the particles on the weighted look and on SURFACE, no GL error; and
  depth_velocity's output (its `output_fbo`, the z velocity) compared with the code before F1, a scratch `git
  archive` of HEAD run the same way (no checkout), on 20261112-125832_measure over the same 20 s from 10 s in: its
  mean, p99 and the misses' share, within the spread of two runs of the code before; the motion's monitor by eye if
  the runs do not line up.

## G. Minor fixes found here, in flow's code

Each gets the flow thread's feedback after its commit.

- G1. `ParticleSurface`'s unused `uOutputSize` goes: its declaration in `flow/fluid_3d/shaders/particle_surface.frag`,
  the `glUniform2f` and the `output_size` parameter of `ParticleSurface.use`, and the argument
  `particle_density._composite_surface` passes. The Fluid3D thread's fix of the surface's third target goes first,
  on the current code (the third target and its clear); G1 and F1 build on top of it, and nothing here touches
  `particle_density.py` before that commit is in. Why: the fragment shader declares it and reads it nowhere, the vertex
  shader does not have it, so the program leaves it out and the set warns (seen, D2's deep_flow -sim, 2026-11-21).
  Check: pyright, flow's tests, deep_flow -sim with the particles on SURFACE, no GL error.
- G2. The bridge's `WeightedSum` (`flow/bridge_3d/shaders/weighted_sum.py` and `.frag`) and `WeightedSum2D`
  (`flow/bridge_2d/shaders/weighted_sum_2d.py` and `.frag`) are renamed `SourceMix` and `SourceMix2D`, class and
  files (a shader's source is found by its module's name) and every call site, and `BRIDGE_2D.md` and `BRIDGE_3D.md`
  name them so; `velocity_bandpass*.py`, `flow_util.py` and hd_trio's `smooth_trail.py` use gl's `shaders.WeightedSum`
  and stay. Why: they mix the bridge's sources (up to seven, a weight each, a source as its red or its length, the sum
  capped if asked), another operation than `gl.shaders.WeightedSum` (two images, each times its weight) under the
  same name (seen); `shaders.Mix` is taken. Check: pyright, flow's tests, deep_flow -sim (the 3D bridge) and low_flow
  -sim (the 2D bridge), no GL error.

Form, done only in a commit that opens the file anyway, never as a step of its own: PEP 8 in `fbo.py` (the one-line
`if`, the double spaces before `->`, a class docstring, the end-of-file newline); the two swap tests become one
`test_swap.py` in a commit that changes `Swap` or them.

## Rules this plan touches

- CLAUDE.md: Inside a package, code that does the same is written once
- CLAUDE.md: Use Python 3.12 and follow PEP 8
- CLAUDE.md: Similar things share the noun and differ by a qualifier (`Frame`, `CameraFrame`)
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change
- CLAUDE.md: A thread that finds a minor problem fixes it, and asks the thread the code belongs to for feedback
- CLAUDE.md: A plan step that is checked by a run or a test is one commit
- gl.md: A texture is held in GL's row order
- gl.md: Shaders get checked in a GPU harness
- gl.md: Name a shader's `.frag`/`.vert`/`.comp` source after the `.py` file that holds its class, beside it
- gl.md: Keep a swap pair itself, never one of its buffers; read `current` and `back` where they are used
- docs.md: When renaming or removing a section, state or setting, search the documents for references to it
