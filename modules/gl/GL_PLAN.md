# GL plan

The package held to the rules and to the textbook layout, from the structure review of 2026-11-20. Work order: I5,
I7: I5 moves the files after their edits. The checks that run the GPU wait for a time Matthias is away from the PC; a
step with such a check is committed after it passes.

## I. The package's structure

`modules/gl` holds the window side, text, two effects and helpers beside its GL resources, flat in its root, and writes
the texture classes and their ping-pong pairs three times each. The target: `gl` holds GL (context, textures,
buffers, programs, the blend state, GL debugging), and the window side is a package of its own, as Cinder's `app`
is. gl's loose files are grouped in folders by job; that comes from CLAUDE.md's package rules, not from Cinder,
whose `gl` is flat.

- I5. gl's loose files in folders by job: `texture/` (`texture_base`, `swap`, `texture`, `texture_3d`,
  `texture_2d_array`, `texture_format`, `fbo`, `layered_fbo`, `image`, `tensor`, and `cudart`, its CUDA side),
  `shader/` (`shader`, `compute_shader`, `file_watcher`, and `storage_buffer`, the compute shaders' buffer),
  `state/` (`context`, `style`, `clear`) and `debug/` (`gl_debug`, `gl_state`, `gpu_timer`); `shaders/` stays the
  library of operations; `colormap` goes to `shaders/`
  beside `colormap.glsl`. The tests stay in `gl/tests`. `from modules.gl import …` stays for every
  name. Check: pyright, the guard tests, gl's tests, each app's main imported.
- I7. `Style` becomes `BlendState` (GL.md's and Cinder's word), everywhere: the module `style.py`, its `Style`
  import alias, `StyleState`, `StyleStack`, `push_style`/`pop_style`, `Context`'s stack, `test_style`, their users
  and GL.md; the rules thread updates gl.md's line on it and adds `BlendState` (not `Style`) to the glossary in the
  same work, and the coordinator is told when it is built. Check: pyright, the guard tests, gl's tests.

## Rules this plan touches

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
- glossary.md: `BlendState`, not `Style` (I7, by the rules thread)
- CLAUDE.md: Apps and modules import another package from its `__init__.py`, never from its internal files
- CLAUDE.md: Inside a package, its subpackages included, import relatively; tests may import absolutely
- CLAUDE.md: Name a concept after what it is in the domain, and use that one term everywhere it appears
- CLAUDE.md: Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change
- CLAUDE.md: Update all affected call sites in the same change
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: A commit leaves no pyright errors
