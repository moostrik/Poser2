# GL plan

The package held to the rules and to the textbook layout, from the structure review of 2026-11-20.

## I. The package's structure

`gl`'s blend state is called `Style`, where GL.md and Cinder call it the blend state.

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
