# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## A. One rule per bullet

- A1. Go through the CLAUDE.md bullets that hold more than one rule, one at a time, with Matthias

## B. The settings rule files

The settings review found lines in `.claude/rules/settings-presets.md` and `settings-backend.md` that are wrong, cannot be checked, or are said elsewhere. What is left of `settings-backend.md` fits in `CLAUDE.md`, so the file goes. `settings-presets.md` stays until the settings package's document (the settings plan) holds what it describes. Rule text only, one commit.

- B1. `settings-presets.md`: its two lines under "Rules that go" leave; the heading "Composition over inheritance" becomes "Sharing"
- B2. `settings-backend.md` is deleted; the two lines under "New rules" go into `CLAUDE.md`; a search for `settings-backend` finds what still points at the file

### Rules that go

From `settings-presets.md`:

- "`access=Field.INIT` fields are in the JSON but skipped by `update_from_dict()` — still keep them for documentation". Wrong: the code loads them until `initialize()` (seen in `base_settings.py`)
- "Do not subclass a module's Settings class in app code — use `Group(ModuleSettings, share=[...])` instead". 23 app classes do and none can be a `Group` with `share` (read by the settings thread); CLAUDE.md's "Prefer composition over inheritance" holds what is left of it

`settings-backend.md`, the whole file:

- "Maintain the settings engine contract and reactive behavior in `modules/settings/`". Cannot be checked
- "Preserve descriptor contracts for `Field`, `Group`, and `FieldAlias`". Cannot be checked
- "Keep `initialize()`, `update_from_dict()`, and `to_dict()` semantics stable and documented". Becomes the new line in CLAUDE.md Settings
- "Maintain thread-safety guarantees around locks and callback dispatch". Said by CLAUDE.md Concurrency
- "Keep callback fan-out fault-tolerant so one failure does not stop other callbacks". Becomes the new line in CLAUDE.md Concurrency
- "Preserve shared field propagation behavior: bidirectional sharing via `share=` (parent↔child)". Said by `settings-presets.md`
- "Preserve `.as_()` alias mapping behavior for child constructor names while serialization stays parent-keyed". Cannot be checked
- "Keep JSON serialization/deserialization deterministic". Cannot be checked
- "Treat GUI metadata as hint-only data; backend logic must never depend on GUI frameworks". Held by the settings package's test (read by the refactor thread)
- The scope note. Says where other rules are

### New rules

- `CLAUDE.md` Concurrency, after "A callback list has its own lock; …": "A failing callback does not stop the others"
- `CLAUDE.md` Settings, as its last line: "Prefer not to change what `initialize()`, `update_from_dict()` and `to_dict()` do; every app's presets depend on it"

## C. The Waker

The refactor plan builds `utils.Waker` (its B1), which replaces `utils.wait_until`: a thread waits on it until a moment, and a new item or stop wakes it with `set()`. Rule text only, one commit, when the Waker is committed and before the loops are moved to it.

- C1. `CLAUDE.md` Time: the line that goes makes place for the new one
  - goes: "Pace a loop with `utils.wait_until`; a wait's timeout (`Event.wait(t)`, `Queue.get(timeout=t)`) ends on Windows' 15.6 ms tick"
  - new: "A thread that waits for a moment waits on a `utils.Waker`; a wait's timeout (`Event.wait(t)`, `Queue.get(timeout=t)`) ends on Windows' 15.6 ms tick"

## Rules it touches

- `CLAUDE.md` "One rule per bullet": A
- `CLAUDE.md` "A rule describes the codebase as it should look": B
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": B
- `CLAUDE.md` Time: C
