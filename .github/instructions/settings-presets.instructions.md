---
description: "Use when modifying settings schema classes and when schema changes require preset JSON updates."
applyTo: "apps/**/settings.py, modules/**/settings.py, files/settings/**/*.json"
---
# Settings Schema & Preset Workflow

## Intent

Keep preset JSON files synchronized with settings schema changes.

## GUI independence

The settings model must remain GUI-independent.

Rules:
- Field metadata (`description`, `widget`, `options`, `regex`, etc.) are optional hints for GUI clients, not behavioral contracts
- Settings code is GUI-agnostic; all GUI-specific behavior and rendering logic belongs in GUI adapter implementations
- Multiple independent GUI implementations must be able to coexist without changes to settings or consuming code
- Changing Field metadata should never cause runtime behavior changes or require preset updates
- Serialization and deserialization completely ignore GUI metadata; JSON presets are GUI-independent

## Preset update workflow

After renaming, adding, or removing a `Field`, `Group`, or `Child`:

1. Find all `.json` files in `files/settings/<app>/`
2. For **renamed** fields: find the old key in each JSON and rename it to the new key, preserving the value
3. For **added** fields: add the key with the Python `Field` default value
4. For **removed** fields: delete the key from each JSON
5. For **restructured** groups (moved/renamed): restructure the corresponding JSON object

## Key rules

- JSON keys must match Python field names exactly (after `share` aliasing)
- `access=Field.INIT` fields are in the JSON but skipped by `update_from_dict()` — still keep them for documentation
- Shared fields (`share=[...]`) appear in the **parent** JSON, not the child — the parent propagates values to children at construction
- Fields using `.as_('child_name')` are serialized under the **parent's** field name, not the alias
- `Group` and `Child` entries become nested JSON objects; the key is the Python attribute name

## Scope note

This file governs settings schema and preset JSON maintenance.
Settings backend internals in `modules/settings/` are governed by
`.github/instructions/settings-backend.instructions.md`.
