# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred.

## A. Names

- A1. `StagingRing` (`preprocessing/staging_ring.py`) is renamed after what it is, a pool of staging buffers taken
  in turn (triple buffering), not a ring buffer; the name is decided at its go, and its module, `ImageUpload`'s use
  (`preprocessing/image_upload.py`) and INFERENCE.md change with it. Why: it holds `STAGING_COUNT` separate pinned
  buffers per key, takes the next with `slot = (slot + 1) % STAGING_COUNT` and waits for that buffer's copy event
  (`StagingRing.buffer`); a ring buffer is one block with a wrapping head **(seen)**. Check: the inference unit
  tests, an import of each app's main.

## The rules it touches

- `CLAUDE.md` Naming: name a concept after what it is in the domain, and use that one term everywhere it appears
