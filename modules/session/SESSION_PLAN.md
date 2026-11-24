# Session plan

Found by the utils scan of 2026-11-22, for a later thread; nothing here is decided or built. Each finding is
**(seen)** in the code, or **(deduction)**. The users were read on GitHub's deep-flow of 2026-11-20 and may have moved
since.

## Questions

Q1. `modules/session` holds two unrelated things in three files, so the package goes:
- `Session`, the recordings' chunk timer, moves to `recording` as `session.py` at its root; recording's
  `__init__.py` exports `Session` and `SessionSettings`, and the apps' mains and settings (deep_flow, low_flow,
  white_space, hd_trio) import them from there. The pose recorder hears the split through settings only, so pose gets
  no recording import (recording's feedback).
- `SequencerState` moves to `inout` beside `osc_sound`. App code makes it (hd_trio's sequencer, white_space's state
  machine) and inout's `osc_sound` reads it as its wire format **(seen)**; `modules/board` has no sequencer slice, only
  hd_trio's own `board/sequencer_state.py` **(seen**, PC listing**)**, and board.md has a slice import its type from
  the package it is about (Module Refactor's feedback).
- `session/settings.py` goes: its one line is a comment holding a literal `\n`, so it re-exports nothing **(seen)**.
Why: CLAUDE.md Packages "A package is one thing, and its name is that thing".

Q2. `Session` binds its start and stop fields and never unbinds them, and it has no stop to do it in **(seen**,
recording's feedback**)**. A stop that unbinds them, after the move.
Why: CLAUDE.md Settings "Components that `bind()` must `unbind()` the same callbacks on teardown".

Its chunk timer's plain `Event.wait` stays: UTILS.md lists it among the timers on a plain wait, a split at most one
tick late.

## Rules this plan touches

- Packages: A package is one thing, and its name is that thing
- Settings: Components that `bind()` must `unbind()` the same callbacks on teardown
