---
description: "Use when modifying, adding, or removing Field, Group, or Child definitions in any app settings file (apps/*/settings.py), or when editing preset JSON files in files/settings/."
applyTo: "apps/*/settings.py, files/settings/**/*.json"
---
# Settings & Preset Maintenance

## Architecture

Each app has a **settings tree** defined in `apps/<app>/settings.py` using `BaseSettings` subclasses with `Field`, `Group`, and `Child` descriptors. The root class is named `Settings`.

Preset JSON files in `files/settings/<app>/` mirror the settings tree exactly. The startup preset (default: `studio.json`) is loaded via `presets.load()` which calls `update_from_dict()`.

## When changing settings Python code

After renaming, adding, or removing a `Field`, `Group`, or `Child`:

1. Find all `.json` files in `files/settings/<app>/`
2. For **renamed** fields: find the old key in each JSON and rename it to the new key, preserving the value
3. For **added** fields: add the key with the Python `Field` default value
4. For **removed** fields: delete the key from each JSON
5. For **restructured** groups (moved/renamed): restructure the corresponding JSON object

## Composition over inheritance

- Do not subclass a module's Settings class in app code — use `Group(ModuleSettings, share=[...])` instead
- To propagate fields to multiple children, declare them on the parent group and `share` to each child Group
- All sharing is bidirectional: parent writes propagate to children, child writes propagate upward to the parent (and fan out to siblings)

## Key rules

- JSON keys must match Python field names exactly (after `share` aliasing)
- `access=Field.INIT` fields are in the JSON but skipped by `update_from_dict()` — still keep them for documentation
- Shared fields (`share=[...]`) appear in the **parent** JSON, not the child — the parent is the serialization source of truth
- Fields using `.as_('child_name')` are serialized under the **parent's** field name, not the alias
- `Group` and `Child` entries become nested JSON objects; the key is the Python attribute name
