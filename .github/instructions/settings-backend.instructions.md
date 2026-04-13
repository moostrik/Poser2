---
description: "Use when editing settings backend internals (descriptors, lifecycle, serialization, and propagation) in modules/settings/."
applyTo: "modules/settings/**/*.py"
---
# Settings Backend Internals

## Intent

Maintain the settings engine contract and reactive behavior in `modules/settings/`.

## Backend rules

- Preserve descriptor contracts for `Field`, `Group`, `Child`, and `FieldAlias`
- Keep `initialize()`, `update_from_dict()`, and `to_dict()` semantics stable and documented
- Maintain thread-safety guarantees around locks and callback dispatch
- Keep callback fan-out fault-tolerant so one failure does not stop other callbacks
- Preserve shared field propagation behavior: push (parent→child), pull (child→parent), bidirectional (both push and pull)
- Preserve `.as_()` alias mapping behavior for child constructor names while serialization stays parent-keyed
- Keep JSON serialization/deserialization deterministic and backward-compatible with existing presets
- Treat GUI metadata as hint-only data; backend logic must never depend on GUI frameworks

## Consumer usage rules

- Pass settings (or `Group`/`Child` subgroups) via constructor dependency injection, not globals
- Bind reactive updates explicitly with callbacks and keep callback logic thread-safe
- Cache `Field.INIT` values when useful; do not long-term cache mutable non-INIT values
- Avoid circular updates where a callback writes the same field that triggered it
- Treat READ-only values as snapshots; copy arrays before cross-thread handoff

## Scope note

This file governs the settings system implementation in `modules/settings/`.
Consumer usage and preset workflow rules are defined in
`.github/instructions/settings-presets.instructions.md`.
GUI-specific guidance may be split into a dedicated third file later.