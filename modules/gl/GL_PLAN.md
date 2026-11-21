# GL plan

Fewer GL calls on the main thread, and the package held to the rules and to the textbook layout. G comes from the
Cinder research (2026-11-20), H from the package scan, I from the structure review of the same day. Work order:
I6, G2, I5, I7: the shaders do not move after G1 edits them, G2's bind is written once on I6's base,
and I5 moves the files after their edits. Before
the first step, `shader_check.py --reference` makes the reference images at HEAD; every shader check compares
against them. The checks that run
the GPU (`shader_check.py`, the -sim runs) wait for a time Matthias is away from the PC; a step with such a check is
committed after it passes.

## I. The package's structure

`modules/gl` holds the window side, text, two effects and helpers beside its GL resources, flat in its root, and writes
the texture classes and their ping-pong pairs three times each. The target: `gl` holds GL (context, textures,
buffers, programs, the blend state, GL debugging), and the window side is a package of its own, as Cinder's `app`
is. gl's loose files are grouped in folders by job; that comes from CLAUDE.md's package rules, not from Cinder,
whose `gl` is flat.

- I5. gl's loose files in folders by job: `texture/` (`texture`, `texture_3d`,
  `texture_2d_array`, `texture_format`, `fbo`, `layered_fbo`, `image`, `tensor`, and `cudart`, its CUDA side),
  `shader/` (`shader`, `compute_shader`, `file_watcher`, and `storage_buffer`, the compute shaders' buffer),
  `state/` (`context`, `style`, `clear`) and `debug/` (`gl_debug`, `gl_state`, `gpu_timer`); `shaders/` stays the
  library of operations; `colormap` goes to `shaders/`
  beside `colormap.glsl`. The tests stay in `gl/tests`. `from modules.gl import …` stays for every
  name. Check: pyright, the guard tests, gl's tests, each app's main imported.
- I6. One texture base: `TextureBase` (the GL name, size, format, wrap, filter, bind, clear, read) under `Texture`,
  `Texture3D` and `Texture2DArray`. The pairs subclass what they swap, and callers pass a `SwapFbo` where an `Fbo`
  goes, so a generic `Swap[T]` holds the pair and the swap, and `SwapFbo(Swap[Fbo], Fbo)`, `SwapTexture3D` and
  `SwapTexture2DArray` stay as thin classes over it. Check: tests for clear, read and swap, and `shader_check.py`.
- I7. `Style` becomes `BlendState` (GL.md's and Cinder's word), everywhere: the module `style.py`, its `Style`
  import alias, `StyleState`, `StyleStack`, `push_style`/`pop_style`, `Context`'s stack, `test_style`, their users
  and GL.md; the rules thread updates gl.md's line on it and adds `BlendState` (not `Style`) to the glossary in the
  same work, and the coordinator is told when it is built. Check: pyright, the guard tests, gl's tests.

## G. Fewer GL calls

Standard GL 4.5, not from Cinder. Measured (G0, 2026-11-20, -sim, 20 s from 10 s): white_space makes 1000 GL calls a
frame, 72 of them sampler units and 153 texture binds; deep_flow 2338, 509 sampler units and 608 binds. One PyOpenGL
call costs 0.4 to 0.7 µs on the main thread, so all of them take about 0.4 ms (white_space) and 1.3 ms (deep_flow) a
frame (deduction). G1 and G2 are built as the textbook way, G1 after I4 and G2 after I6. After G2, one -sim run
per app (about 45 s each) checks G1, I6 and G2 together: at each draw and dispatch every sampler the program reads
has a texture at its unit, the GL calls a frame against G0's, and no GL error in the log; a fault it finds is fixed
in a commit of its own.

- G2. Direct state access: `glBindTextureUnit` for a texture's bind and `glTextureSubImage2D` for its upload; the CUDA
  upload in `tensor.py` from 6 calls to about 3, its `glGetError` behind the debug flag. Every caller of
  `Texture.bind`/`unbind` moves with it. Check: `scripts/shader_check.py`, then the -sim runs above.

## Rules this plan touches

- gl.md: Prefer as few GL calls and state switches as possible
- gl.md: Prefer not to read back from the GPU or wait on it inside a frame
- gl.md: All OpenGL calls execute on the main thread, in the window loop
- gl.md: Name a shader's `.frag`/`.vert`/`.comp` source after the `.py` file that holds its class, beside it; the loader resolves it by that module name
- gl.md: A shader is named after the operation it does; the `shaders` package in front says what it is (`shaders.PosePointLines`)
- gl.md: Shaders get checked in a GPU harness
- gl.md: the line on `Style` (I7, by the rules thread)
- docs.md: A component is described in its own document; other documents link to it
- CLAUDE.md: A package is one thing, and its name is that thing
- CLAUDE.md: Its `__init__.py` holds what other packages use
- CLAUDE.md: Its files each hold one class or one subject, named after it
- CLAUDE.md: A package with many loose files groups them in folders by job
- CLAUDE.md: Prefer no folder for one or two files
- CLAUDE.md: Inside a package, code that does the same is written once
- CLAUDE.md: Its tests sit in its `tests/` folder
- CLAUDE.md: Its shaders sit in its `shaders/` folder
- CLAUDE.md: Downstream of the camera, take time from the frame or the tick instead of reading a clock
- glossary.md: `BlendState`, not `Style` (I7, by the rules thread)
- CLAUDE.md: Apps and modules import another package from its `__init__.py`, never from its internal files
- CLAUDE.md: Inside a package, its subpackages included, import relatively; tests may import absolutely
- CLAUDE.md: Name a concept after what it is in the domain, and use that one term everywhere it appears
- CLAUDE.md: Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change
- CLAUDE.md: Prefer the design with the lower latency, measured in the context of the app
- CLAUDE.md: Update all affected call sites in the same change
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: Run a test of the app or a GPU harness only on the user's go, and only what the plan names
- CLAUDE.md: A commit leaves no pyright errors
