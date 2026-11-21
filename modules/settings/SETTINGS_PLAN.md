# Settings plan

The package scan of 2026-11-20 (modules/settings held up to CLAUDE.md and its rules), with the plan's analysis.

## E. The package held up to the rules

- E4. `widget.py`'s "Adding a widget" goes into SETTINGS.md; the `_propagate_*` docstrings and `Field.fire`'s say what
  each one is.
- E8. The panel polls every value control (`modules/panel/controls.py` `Control.polls()`; buttons and `INIT` fields
  as now), so a value written by code shows whatever its access says; the polling runs in the panel's process, and
  the app's side (`mirror.ValueDiff`) already compares every field. SETTINGS.md's Access section says so. Checked by
  the panel tests.
- E9. One map of a class's fields and groups, built once per class in `BaseSettings.__init_subclass__`:
  `BaseSettings.__init__`, `to_dict` and `Group.validate_wiring` read it, in place of their three walks of the MRO;
  `_get_group_descriptor` goes.
- E10. One enum coercion (a member from its name or value), used by `Field._coerce` and `_coerce_element`.
- E11. `Field.encode` encodes a list's elements through the same function as a value.
- E12. A missing setup file starts the app on its defaults with a warning, as smile does, in place of refusing the
  start: `SettingsStore.load_startup` logs and goes on, and the four mains (deep_flow, low_flow, hd_trio,
  white_space) start on the defaults; the panel's Save then writes the file. Checked by the store's tests.

Each is checked by the settings and panel tests and pyright; no app run.

## Rules this plan touches

- CLAUDE.md: Its `__init__.py` holds what other packages use
- CLAUDE.md: Its files each hold one class or one subject, named after it
- CLAUDE.md: Inside a package, code that does the same is written once
- CLAUDE.md: A docstring says what the thing is
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: Log exceptions with context; never fail silently
- CLAUDE.md: Use threads with explicit synchronization, not async/await
- CLAUDE.md: Fix a problem where it is made, not where it shows
