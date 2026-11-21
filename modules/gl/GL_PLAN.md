# GL plan

What is not built in `modules/gl`. A step is one commit, built on Matthias's go, checked by pyright, the guard
tests, gl's tests and the runs it names; a run happens at its step, after a tasklist check that no app is running,
unless Matthias says he is using the PC or the app.

## E. Every pixel transfer sets its own alignment

- E1. `Tensor._update_with_pbo`, `Tensor._update_with_cpu` and `TextureBase._read` set their own unpack or pack
  alignment at the transfer, as `Texture.upload` does; `flow/motion/solve_capture.read_level` and
  `apps/deep_flow/scripts/view_check.read` set the pack alignment at their read, or read through `TextureBase._read`;
  `flow/bridge/velocity_peak` stays as it is (one row of 256 × R32UI, the same bytes at any alignment). Why: these
  transfers rely on the alignment GL has at the moment, 4 by default or 1 as `Texture.upload` and `Image` leave it,
  so a row whose bytes are not a multiple of 4 (an R8 or RGB8 texture of odd width) comes out sheared at 4 (seen:
  the transfers, a scan of 2026-11-21; the shear, deduction). After it, the render plan's E1 drops
  `ScreenCapture`'s and `ScreenPreview`'s set-back to 4. Check: shader_check.

Form, done only in a commit that opens the file anyway, never as a step of its own: PEP 8 in `fbo.py` (the one-line
`if`, the double spaces before `->`, a class docstring, the end-of-file newline); the two swap tests become one
`test_swap.py` in a commit that changes `Swap` or them.

## Questions

- `gl.shaders.WeightedSum` and `flow.bridge_3d`'s `WeightedSum` are two operations under one name (the flow's is the
  bridge's mix of up to seven sources): the flow's renamed after what it does? The flow thread's call.

## Rules this plan touches

- CLAUDE.md: Inside a package, code that does the same is written once
- CLAUDE.md: Use Python 3.12 and follow PEP 8
- CLAUDE.md: A plan step that is checked by a run or a test is one commit
- gl.md: A texture is held in GL's row order
- gl.md: Shaders get checked in a GPU harness
