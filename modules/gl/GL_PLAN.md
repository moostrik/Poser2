# GL plan

Proposed by the package scan of 2026-11-21, on the code at c5d86a7d, with the rules thread's sanity check and the
plan analysis folded in. The why of each step is the finding it rests on, marked seen (in the code) or deduction.
Every step is one commit and is checked by pyright, the guard tests and gl's tests, and by the runs it names;
shader_check compares with references made at HEAD before the first step. A run happens at its step, after a
tasklist check that no app is running, unless Matthias says he is using the PC or the app.


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
