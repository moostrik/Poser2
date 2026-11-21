# Settings

`modules/settings` holds the reactive settings every app and module is configured with: a tree of `BaseSettings`
classes, each a pure data container whose fields are declared as class attributes, read and written as plain
attributes, and bound to callbacks. The panel (`modules/panel`) is built from the tree; the store saves and loads it.

## Fields

A `Field` is a descriptor declared on a `BaseSettings` subclass. Its type is inferred from its default; a list's
from its first element, so a list default holds at least one. A write is coerced to that type: an int becomes a
float, an enum is made from its name or value, a tuple or list builds a value object (`Color(r, g, b)`), a dict one
with `from_dict`; anything else raises `TypeError`. A bool never passes for an int or a float.

A write that changes the value fires the field's callbacks, outside the field's lock, each in a `try`: a failing
callback is logged and the others still run. A list is handed out as a copy, so a change in place fires nothing;
the whole list is written again.

Every keyword but `default`, `access` and `saved` is a panel hint (`min`, `max`, `step`, `description`, `label`,
`unit`, `row_label`, `width`, `visible`, `pinned`, `newline`, `widget`, `color`, `options`, `drag`, `seek`): the
descriptor neither clamps nor checks against them. `widget` picks the control; `Widget.resolve` picks one from the
type when it is `Widget.default`, and a widget that cannot show the type is refused at declaration.
`saved=False` leaves the field out of the files.

A button (`widget=Widget.button`) is a `Field[bool]` that stays `False`: `fire()` calls its callbacks with `True`,
down to the children it is shared to and up to its parent's field. A button is never saved.

## Access

`access` says who writes a field: whether the panel lets the user edit it, and whether it can change at all.

| Access      | The panel     | Polled | Written by                                   |
|-------------|---------------|--------|----------------------------------------------|
| `READ`      | shows it      | yes    | the code (a frame rate)                      |
| `WRITE`     | edits it      | yes    | the user                                     |
| `READWRITE` | edits it      | yes    | the user and the code (a window's size)      |
| `INIT`      | a fixed label | no     | the files and the constructor, until `initialize()` |

`WRITE` is the default. The panel polls every field but an `INIT` one, so a value the code writes shows whatever its
access says; `READ` and `READWRITE` say the code writes it. `READ` is a hint: the code may write any field. `INIT` is
enforced: after `initialize()` a write raises `AttributeError` and a load skips the field.

## Groups and sharing

A `Group(SettingsClass)` declared on a class makes a child instance of it, reached as an attribute, which cannot be
replaced. Its keywords are panel hints but `defaults` and `share`: `last` puts it after the other children (a
subclass's included), `description` is its panel text, `tint` its panel's color.

`defaults` gives the child's own fields their starting values in this place in place of the class defaults; the
field's `default`, which the panel shows, stays the class's.

`share` lists parent fields whose value the child holds too, under the same name or another (`field.as_('name')`).
The share runs both ways: a write to the parent sets every child it is shared to, and a write to a child sets the
parent, which sets the other children. The two fields must have compatible types and be both `INIT` or both not;
`validate_wiring` raises `TypeError` at construction otherwise. A shared field is saved by its parent, under the
parent's name, and left out of the child's dict.

A subclass lists its base's fields and groups first; a name it declares again keeps the base's place.

## Serialization and `initialize()`

`to_dict()` gives the tree as plain values: each field by `Field.encode` (an enum by its name, a value object as its
dict), each child as a nested dict under its attribute name. It leaves out `READ` fields, buttons, `saved=False`
fields and a child's shared fields; it keeps `INIT` fields.

`update_from_dict(data)` sets the instance's own fields first, so a count field is current before the children
load, then each child's dict, then pushes the shared fields down again. An unknown key and a value that cannot be
coerced are logged and skipped, so one bad key does not stop the rest. `INIT` fields load only before
`initialize()`. A field a file does not hold keeps its default, so a new field's key goes only into the files that need
another value.

An app builds its tree, loads its startup files, sets what it derives (a crop sized from the pose model), then calls
`initialize()`, which locks the `INIT` fields of the whole tree. Every app's files depend on what these three do.

## The store and its root groups

An app's root holds root groups: `setup`, the machine and the site (cameras, windows, panel, links, the pose model,
what the operator screen shows), and in the flow apps `flow`, the flow chain. A value the flow needs too
(`input_fps`) stays a root field shared into both.

`SettingsStore`, made from the app's settings folder, keeps each root group's named files in its own folder:

```
apps/<app>/data/settings/
    .startup.json          the startup file of each root group, {"setup": "studio", "flow": "mattijs"}
    setup/<name>.json      the root's own fields beside "setup"
    flow/<name>.json       the flow group itself
```

`save(root, group, name)` writes a group from `root.to_dict()`, atomically (a temporary file, then a replace); the
setup's file is the root's dict without the other root groups, so a root field keeps its key beside `setup`.
`load(root, group, name)` loads a file through `update_from_dict`, so loading one root group leaves every other field
as it was. A root group not named in `.startup.json` starts on the store's default name (`studio`; smile's `Smile`).
`load_startup(root)` loads the setup first, then the other root groups; a missing setup file stops the start
(`FileNotFoundError`), a missing file of another group is logged and its defaults stay. A name is refused when it is
empty, holds a path separator or starts with a dot.

The panel shows a row for each root group (`modules/panel/file_controls.py`) and a tab for each child of a root
group; a root group declared with `Group(..., after="<card path>")` has no tabs: its children follow that card as
cards in its tab (the flow apps' `flow`, the chain, after `setup.render.layer` in the render panel).

## The mirror

The panel runs in its own process and never imports an app's settings classes, which pull in the app's whole stack.
`mirror.describe(root)` gives the tree as plain values (instances, fields, hints, labels, wiring, each enum by its
members); `mirror.build_mirror` makes a tree of the same shape from it. `mirror.values` and `mirror.apply` carry the
values, encoded as the files hold them and coerced by `Field.set` on arrival; `mirror.ValueDiff` gives the values
changed since the last send. A field is addressed by its instance's path, the child names from the root joined by
dots ("" for the root), and its name (`mirror.walk`, `mirror.find`).

## Measured

The cost of the settings writes (`BaseSettings.__setattr__`, a field's write through it; the part that looks for a
`property` setter in the class's MRO beside it), 2026-11-20, commit 7b787812, deep_flow `-sim` on its startup files
(setup `studio`, flow `mattijs`), 30 s after a 25 s warm-up, 1771 render ticks; timed inside the write, so the
timing adds to it. The most frequent writes are the pinned readouts, each once a tick.

| Measure        | Total     | Per tick  | Per write |
|----------------|-----------|-----------|-----------|
| Writes         | 46 829    | 26.4      |           |
| A write        | 153.3 ms  | 0.087 ms  | 3.27 µs   |
| Its MRO scan   | 54.8 ms   | 0.031 ms  | 1.17 µs   |

Below 0.1 ms a tick, so no cache. A field's write looks in the fields first: the MRO scan runs only for a name
that is no field, so a field's write no longer pays its row (not measured again).
