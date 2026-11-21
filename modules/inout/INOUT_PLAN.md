# Inout plan

What is not built in `modules/inout`, from the package scan of 2026-11-21. **(seen)** is read in the code,
**(deduction)** is inferred.

## B. Clean-up

- B2. `OscSound`'s docstring and slot names. Why: its docstring says it sends "at a configurable frame rate"; it
  sends one bundle per `submit_frames` **(seen)**; a slot is `id` in `_inactive_messages`, `_tick_messages` and
  `OscSoundExtras.inactive_messages`, where the glossary's term is `track_id` **(seen)**. How: the docstring says
  what it is; `id` becomes `track_id` there and in white_space's `SoundExtras`. Check: inout tests, white_space
  `test_osc_sound_poses`, `test_osc_sound_globals`.
- B3. `OscSound` copies less. Why: `submit_frames` copies the frame dictionary, while an item handed to a callback
  is not changed afterwards (`CLAUDE.md`) **(seen; that the pipeline keeps to it is a deduction)**; `_active_messages`
  makes its zero rows for every player on every tick **(seen)**. How: keep the dictionary as handed over; the zero
  rows made once, `max_players` being `Field.INIT`. Check: inout tests, white_space's sound tests.
- B4. `OscReceiverSettings.last_time` goes. Why: it is a `perf_counter()` stamp **(seen)**, a number on the panel
  that means nothing to a reader, and `counter` already shows the activity; nothing else reads it and no settings
  file holds it **(seen)**. How: the field and its write in `OscReceiver._handle` go. Check: inout tests, pyright.

## The rules it touches

- `CLAUDE.md` Docstrings, a docstring says what the thing is: B2
- `.claude/rules/glossary.md`, the id of a tracked person is `track_id`: B2
- `CLAUDE.md` Performance, avoid unnecessary copies: B3
- `CLAUDE.md` API evolution, remove obsolete pathways: B4
