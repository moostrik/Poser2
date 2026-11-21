# GL plan

What is not built in `modules/gl`. A step is one commit, built on Matthias's go, checked by pyright, the guard
tests, gl's tests and the runs it names; a run happens at its step, after a tasklist check that no app is running,
unless Matthias says he is using the PC or the app.

## H. One framebuffer class

- H1. After F1. `Fbo` and `MultiTargetFbo` become one `Fbo` with one or more color targets it owns, as Cinder's
  `gl::Fbo`: `allocate(width, height, formats)`, a format per target, target 0 the texture the `Fbo` is (so every
  caller that draws it, binds it or passes it as a texture keeps working), the others by index; `SwapFbo` and
  `SwapMultiTargetFbo` become one swap pair of it, with F1's `target(i)`. Every caller moves with it: the ones with
  one target change only if they name its format by keyword (93 files name `Fbo` or `SwapFbo`: 35 in the apps, 53 in
  the modules, 5 in the scripts), the ones with several (depth_velocity, particle_density) take the one class.
  `LayeredFbo` stays as it is: it is one target of another texture kind (a `Texture2DArray`), attached with all its
  layers (`glFramebufferTexture`) and drawn per layer by `gl_Layer`, so one `Fbo` would have to become generic over
  the texture kind for it (deduction). Why: two classes for one thing, a framebuffer that owns its color targets,
  differing only in their count; the textbook class is one (Cinder's `gl::Fbo` with `Fbo::Format`'s color textures)
  (seen: `fbo.py`, `multi_target_fbo.py` after F1). Check: pyright, the guard and gl tests, shader_check, and a -sim of
  each app (deep_flow with the particles on the weighted look and on SURFACE, low_flow, hd_trio, white_space,
  smile), no GL error.

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
