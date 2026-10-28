---
description: Settings backend internals — descriptors, lifecycle, serialization, and propagation
paths:
  - "modules/settings/**/*.py"
---
# Settings Backend Internals

## Intent

Maintain the settings engine contract and reactive behavior in `modules/settings/`.

## Backend rules

- Preserve descriptor contracts for `Field`, `Group`, `Child`, and `FieldAlias`
- Keep `initialize()`, `update_from_dict()`, and `to_dict()` semantics stable and documented
- Maintain thread-safety guarantees around locks and callback dispatch
- Keep callback fan-out fault-tolerant so one failure does not stop other callbacks
- Preserve shared field propagation behavior: bidirectional sharing via `share=` (parent↔child)
- Preserve `.as_()` alias mapping behavior for child constructor names while serialization stays parent-keyed
- Keep JSON serialization/deserialization deterministic and backward-compatible with existing presets
- Treat GUI metadata as hint-only data; backend logic must never depend on GUI frameworks

## Scope note

This file governs the settings system implementation in `modules/settings/`.
Consumer usage rules are in `CLAUDE.md`; preset workflow rules are in @.claude/rules/settings-presets.md.
