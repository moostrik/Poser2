---
description: Board protocols, mixins, and app render board composition
paths:
  - "modules/board/**"
  - "apps/*/board/**"
  - "modules/render/layers/**"
---
# Board

## Package

- Each file holds one protocol and its mixin
- `__init__.py` exports every protocol and mixin
- `modules/board/` holds what a module or more than one app reads; a slice only one app reads sits in `apps/<app>/board/`, as the same pair

## Protocol and mixin

- A protocol holds the read and write signatures only
- A mixin is self-contained: its own lock, its own storage, its own methods
- Prefix the lock and the storage with the capability name (`_frame_lock`, `_frames`)
- Keep method bodies minimal: get or set under the lock, plus any change of storage layout (the window pivot)
- Import a type that is only named in annotations under `TYPE_CHECKING`

## The app's board

- `apps/<app>/board/render_board.py` defines `class RenderBoard(Mixin1, Mixin2, ...)`
- Its `__init__` calls each mixin's `__init__`; it has no other methods

## Consumer typing

- App code type-hints the concrete `RenderBoard`
- Module code type-hints the narrowest protocol (`HasFrames`, `HasWindows`), never the `RenderBoard`
- A compositor that only passes the board on to its children may use `Any`
