# Settings plan

Today a preset is the whole settings tree (`presets.save`/`load` on the root), so switching it also switches the
camera, the windows and the other machine settings. The settings are saved as config groups (Hydra's structure,
not its library): **every root group has its own folder of named files**, `data/settings/<group>/<name>.json`,
loaded and saved on its own. Each app's root holds `setup` (the machine and the site: cameras and their
exposure, windows, panel, OSC, recorder, pose model, what the operator screen shows) and, in the flow apps,
`flow` (the flow chain). A file of one root group cannot reach another. A later split, the camera on its own,
is `camera` moved from `setup` to the root and its files split once. The word "preset" goes.

## A. The root groups in each app

- A1. The panel's tabs: today each child of the root gets a tab (`page.py` `tab_entries`); they become the
  children of each root group, grouped under it (Setup: camera, pose, render…; Flow: view, color…), so the
  panel keeps its tabs when the root holds only `setup` and `flow`. Until every app has moved, a root without a
  `setup` group keeps one tab per child; A6 removes that path. Test: the panel's tests, a tab per child of a
  root group, and a root without `setup` as today; its run is A2's.

Then each app, one step each: move the app's groups under its root groups, in the order the code runs them; update every call
site (`settings.camera` → `settings.setup.camera`, `settings.render.fluid` → `settings.flow.fluid`) and the
constructors that take a group (deep_flow's and low_flow's renderer takes `setup.render` and `flow`); restructure
every settings file to match, still whole trees; migrate the panel's `.ui_state.json` keys (its open cards are
keyed by path). The root's own fields move into `setup` and are shared from there; only what `flow` needs too
(`input_fps` in the flow apps) stays at the root and is shared into both. Where GL B2 built `windows`, it moves into `setup`. Run: the app starts on its startup file and
looks as before.

| App         | `flow` holds                       | `setup` holds                                                                     |
|-------------|------------------------------------|-----------------------------------------------------------------------------------|
| deep_flow   | view, color, motion, bridge, fluid | camera, cadence, pose, render (layers, debug, monitors), screen_recorder, inout, panel, windows |
| low_flow    | view, color, motion, bridge, fluid | as deep_flow                                                                      |
| hd_trio     | (none)                             | every root group                                                                  |
| white_space | (none)                             | every root group                                                                  |
| smile       | (none)                             | every root group                                                                  |

A stage's own monitors (`monitor_main`, `monitor_debug`) sit in the stage's settings, so they go with the flow.

- A2. deep_flow.
- A3. low_flow.
- A4. hd_trio.
- A5. white_space.
- A6. smile; the panel's one-level tabs go.

## B. The store and its files

- B1. `presets.py` becomes `store.py`: a `SettingsStore` made from the app's settings folder, in place of the
  module-level `set_app`. The five mains make it and hand it to the panel server; the panel process, being
  another process, makes its own from the folder the server sends; smile's `save/record.py` (`--preset`
  becomes `--setup`, `to_preset` becomes `to_setup`) and every other `set_app` caller (a grep) go over. Its
  imports go to the top. A root group's file is written from `root.to_dict()[group]` (so a field shared down
  stays its parent's) and loaded as `root.update_from_dict({group: data})`; the setup's file is `root.to_dict()`
  without the other root groups, so the root's fields (`input_fps`) keep their key beside `setup` and cannot
  clash with it. `scan`, `path`, `load` and `save` take the root group; `.startup.json` (was `.ui_preset.json`)
  names each root group's startup file, and without it each starts on its default name (smile's setup
  `Smile`, every other `studio`, as `set_app` gives them today); `SettingsStore.load_startup(root)` loads `setup`, then the others. A
  missing setup file stops the start, as a missing preset does today; a missing file of another root group is
  logged and its defaults stay. `to_dict()` and `update_from_dict()` stay as they are. Every app's files move in
  this step: deep_flow setup `studio` (from mattijs, whose camera exposure it keeps) and its 5 flows; low_flow
  setup `studio`, flow `studio`; the other three apps' files into `setup/`. Test: save and load each root group;
  loading one leaves every other field as it was. Run: each of the five apps starts on its startup files.
- B2. The panel's header: a row (star, dropdown, reload, save, save as, delete) for each root group, labelled
  by it ("Setup", "Flow"); the panel's `PresetStore` becomes `SettingsStore` and its load and save, and the
  server's, carry the root group. Test: the panel's tests; run: deep_flow, a setup and a flow saved and loaded.

## C. Documents and rules

- C1. `modules/settings/SETTINGS.md`: fields, access, groups and sharing, `Group` defaults, serialization and
  `initialize()`, the store and its root groups, the mirror; the docstrings slimmed to what each thing is, their
  stale examples gone (`class CameraSettings(Settings)`, `FieldAlias`'s `InterpolatorSettings(share=...)`,
  `Field`'s "the one keyword with a runtime effect").
- C2. settings-presets.md: what SETTINGS.md now holds goes (the Rules thread is told); glossary: "setup",
  `SettingsStore` and a root group's file, not "preset"; CLAUDE.md's two lines that say "presets" (the
  `to_dict()` line and "Keep presets and settings schemas in sync") say "settings files". Wording proposed to
  Matthias first.

## D. The package held up to the rules (from the module review, SETTINGS_REVIEW.md)

- D1. The tracker and pose compositors write their child renderers' settings every tick
  (`_follow_settings`); settings connect the user and a component. The compositor hands the values to its
  renderers as inputs; the renderers' settings keep only what the user sets. Checked by render's tests and one
  deep_flow `-sim` start.
- D2. `bind_all` without `unbind_all` in hd_trio's `flow/gradient_flow.py` and pose's `window/window_node.py`:
  unbind on teardown. Checked by their tests.
- D3. `Access.WRITE` is the default, so the panel stops polling fields only the user writes; a field the code
  writes says `READ` or `READWRITE`, found by a grep of writes. Checked by settings' and the panel's tests and
  each app's start, the panel's values moving where code writes them.
- D4. `shared_source` and `BaseSettings.parent` have no caller: removed.
- D5. `BaseSettings.__setattr__` scans the class's MRO on every write (about 1.5–2 k a second in deep_flow,
  deduced). Measured first in one deep_flow `-sim` run; the property names cached once only if the writes take
  over 0.1 ms a tick.

D1 and D2 change code in render, pose and hd_trio; they are here because they break settings rules and came
from this package's review.

## Order with the GL plan

GL B2 built the root `windows` group (seen in deep_flow's settings); A2 to A6 move it into `setup`, and B1 splits
its keys into the setup files with the rest.

## Rules this plan touches

- CLAUDE.md: Name a concept after what it is in the domain, and use that one term everywhere it appears
- CLAUDE.md: Use no synonyms for one concept across modules
- CLAUDE.md: Prefer not to change what `initialize()`, `update_from_dict()` and `to_dict()` do; every app's presets
  depend on it
- CLAUDE.md: Keep presets and settings schemas in sync with code changes
- CLAUDE.md: Update all affected call sites in the same change
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
- CLAUDE.md: Pass collaborators and fixed structure (counts known at startup) through constructors
- CLAUDE.md: Settings connect the user and a component, not two components
- CLAUDE.md: Components that `bind()` must `unbind()` the same callbacks on teardown
- CLAUDE.md: Prefer the design with the lower latency, measured in the context of the app
- CLAUDE.md: A commit leaves no pyright errors
- settings-presets.md: For restructured groups (moved/renamed), restructure the corresponding JSON object
- CLAUDE.md: Similar things share the noun and differ by a qualifier
- panels.md: A settings class mirrors the code it configures: a group for each step, in the order the code runs them
- app-lifecycle.md: App code should not leak app-specific policy into shared modules
