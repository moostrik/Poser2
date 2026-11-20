# Settings plan

A to D are built (SETTINGS.md describes them); the rule wording waits for Matthias and the Rules thread.

## C. Rules

- C2. settings-presets.md: what SETTINGS.md now holds goes, and its paths glob (`apps/*/data/settings/*.json`)
  matches no file since the files moved into root-group folders; proposed: renamed `settings-files.md`, holding only
  the steps that keep a root group's files in step with its classes. Glossary: "setup" (`setup`, `SetupSettings`)
  and "settings file" (`SettingsStore`, `data/settings/<group>/<name>.json`), not "preset". CLAUDE.md's two lines
  that say "presets" (the `to_dict()` line and "Keep presets and settings schemas in sync") say "settings files".
  Wording proposed to Matthias and sent to the Rules thread for feedback.

## Questions

- "preset" is left in about 480 docstrings and comments across about 90 code files and about 40 documents: a
  sweep, and in which step? Banning it in the glossary guard (`tests/test_glossary.py`) flags them all.
- hd_trio's render writes `settings.fluid.density.fade_time` every tick from the motion
  (`apps/hd_trio/render/hd_trio_render.py`): a component writing settings it does not own, as D1's compositors did.
- `scripts/bridge_compare.py`'s `depth_range()` reads a `depth` group no tree has (`fluid_compare.py` and
  `motion_bridge_pass_check.py` import it): removed, or read from the view crop's `near` and `far`?

## Rules this plan touches

- CLAUDE.md: Prefer not to change what `initialize()`, `update_from_dict()` and `to_dict()` do; every app's presets
  depend on it
- CLAUDE.md: Keep presets and settings schemas in sync with code changes
- CLAUDE.md: Use no synonyms for one concept across modules
- settings-presets.md: For restructured groups (moved/renamed), restructure the corresponding JSON object
- glossary.md: the names "setup" and "settings file"
