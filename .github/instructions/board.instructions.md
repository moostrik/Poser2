---
description: "Use when editing board protocols, mixins, or app render board composition."
applyTo: "modules/board/**, apps/*/render_board.py, modules/render/layers/**"
---
# Board Guidelines

## Package structure

`modules/board/` is a package. Each file contains one protocol + one mixin pair.
`__init__.py` re-exports all protocols and mixins.

## Adding a new capability

1. Create a new file with the protocol and mixin
2. The mixin owns its own `Lock` — never share locks between mixins
3. Add re-exports to `__init__.py`
4. Add the mixin to whichever app `RenderBoard` classes need it

## Protocol rules

- Protocols define the public read/write contract only (method signatures)
- Use `from __future__ import annotations` and `TYPE_CHECKING` for annotation-only imports
- Protocols live in `modules/board/` so module layers can import them without depending on app code

## Mixin rules

- Each mixin is self-contained: own lock, own storage attributes, own methods
- Use `TYPE_CHECKING` for types used only in annotations; use runtime imports for types constructed in `__init__`
- Prefix lock and storage attributes with the capability name (`_frame_lock`, `_frames`) to avoid collisions
- Keep method bodies minimal — get/set with lock, plus any storage-layout transformation (e.g. the window pivot)

## App render board

- `apps/*/render_board.py` defines `class RenderBoard(Mixin1, Mixin2, ...):`
- `__init__` calls each mixin's `__init__` explicitly
- App render boards have no additional methods — they are pure mixin composition
- App renders and main type-hint the concrete `RenderBoard`; module layers type-hint the protocol they need

## Consumer typing

- Module layers import and type-hint the narrowest protocol: `HasFrames`, `HasWindows`, etc.
- Never type-hint the concrete `RenderBoard` in module code — that would couple modules to apps
- Compositors that only pass the board through to children without calling methods on it may use `Any`
