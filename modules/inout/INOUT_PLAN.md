# Inout plan

What is not built in `modules/inout`, from the package scan of 2026-11-21. **(seen)** is read in the code,
**(deduction)** is inferred.

## B. Clean-up

- B3. `OscSound` copies less. Why: `submit_frames` copies the frame dictionary, while an item handed to a callback
  is not changed afterwards (`CLAUDE.md`) **(seen; that the pipeline keeps to it is a deduction)**; `_active_messages`
  makes its zero rows for every player on every tick **(seen)**. How: keep the dictionary as handed over; the zero
  rows made once, `max_players` being `Field.INIT`. Check: inout tests, white_space's sound tests.
- B4. `OscReceiverSettings.last_time` goes. Why: it is a `perf_counter()` stamp **(seen)**, a number on the panel
  that means nothing to a reader, and `counter` already shows the activity; nothing else reads it and no settings
  file holds it **(seen)**. How: the field and its write in `OscReceiver._handle` go. Check: inout tests, pyright.

## The rules it touches

- `CLAUDE.md` Performance, avoid unnecessary copies: B3
- `CLAUDE.md` API evolution, remove obsolete pathways: B4
