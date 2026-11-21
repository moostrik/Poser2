# Inout plan

What is not built in `modules/inout`, from the package scan of 2026-11-21. **(seen)** is read in the code,
**(deduction)** is inferred.

## B. Clean-up

- B4. `OscReceiverSettings.last_time` goes. Why: it is a `perf_counter()` stamp **(seen)**, a number on the panel
  that means nothing to a reader, and `counter` already shows the activity; nothing else reads it and no settings
  file holds it **(seen)**. How: the field and its write in `OscReceiver._handle` go. Check: inout tests, pyright.

## The rules it touches

- `CLAUDE.md` API evolution, remove obsolete pathways: B4
