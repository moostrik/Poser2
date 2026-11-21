# Settings plan

The package scan of 2026-11-20 (modules/settings held up to CLAUDE.md and its rules), with the plan's analysis.

## E. The package held up to the rules

- E4. `widget.py`'s "Adding a widget" goes into SETTINGS.md; the `_propagate_*` docstrings and `Field.fire`'s say what
  each one is.
- E13. A shared button runs each callback once per press, wherever it is pressed: a press on a child goes up to the
  parent and from there down to the other children, not back down into the child it came from, which today runs
  its callbacks twice. Test: a press on a child and a press on the parent each run every callback once; checked by
  the settings tests and the tests of deep_flow, low_flow and hd_trio, whose session buttons are shared.

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
